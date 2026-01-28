#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据改写脚本
处理token数超过限制的数据，通过Gemini API改写过长的observation和gpt回答
"""

import json
import os
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from loguru import logger
import tiktoken
import aiohttp
import re

# Qwen API配置（vLLM兼容）
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://125.122.38.32:8027")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
QWEN_MODEL_NAME = "/data/models/Qwen3-8B"
QWEN_API_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"

# Token计数器
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4使用的编码


def count_tokens(text: str) -> int:
    """计算文本的token数"""
    return len(tokenizer.encode(text))


def calculate_data_tokens(data: Dict[str, Any]) -> int:
    """
    计算一条训练数据的总token数
    包括：system + tools + time + 所有conversations中的value
    """
    total_tokens = 0
    
    # 计算system的token
    if "system" in data:
        total_tokens += count_tokens(data["system"])
    
    # 计算tools的token
    if "tools" in data:
        tools_str = json.dumps(data["tools"], ensure_ascii=False) if isinstance(data["tools"], (list, dict)) else str(data["tools"])
        total_tokens += count_tokens(tools_str)
    
    # 计算time的token
    if "time" in data:
        total_tokens += count_tokens(data["time"])
    
    # 计算所有conversations中value的token
    if "conversations" in data:
        for conv in data["conversations"]:
            if "value" in conv:
                total_tokens += count_tokens(conv["value"])
    
    return total_tokens


def get_last_observation(conversations: List[Dict]) -> Optional[tuple]:
    """
    获取最后一个observation及其索引
    返回：(index, observation_value, observation_tokens)
    """
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("from") == "observation":
            obs_value = conversations[i]["value"]
            obs_tokens = count_tokens(obs_value)
            return (i, obs_value, obs_tokens)
    return None


def check_two_round_function_call_structure(conversations: List[Dict]) -> Optional[tuple]:
    """
    检测两轮function_call结构：
    1. 第一个function_call应该是retrieval_tool
    2. 第一个observation返回工具列表
    3. 第二个function_call是实际调用的工具
    4. 第二个observation是工具返回的结果
    
    返回：(first_obs_index, first_obs_value, second_func_call_index, second_func_call_tool_name)
    如果不符合结构，返回None
    """
    func_calls = []
    observations = []
    
    # 收集所有function_call和observation的索引
    for i, conv in enumerate(conversations):
        if conv.get("from") == "function_call":
            func_calls.append(i)
        elif conv.get("from") == "observation":
            observations.append(i)
    
    # 需要至少2个function_call和2个observation
    if len(func_calls) < 2 or len(observations) < 2:
        return None
    
    # 检查第一个function_call是否是retrieval_tool
    # retrieval_tool的调用格式可能是：
    # 1. "query": "...", "source_filter": "toollist"
    # 2. {"name": "retrieval_tool", ...}
    first_func_call_value = conversations[func_calls[0]].get("value", "")
    is_retrieval_tool = (
        "retrieval_tool" in first_func_call_value or 
        "source_filter" in first_func_call_value or
        ("query" in first_func_call_value and "toollist" in first_func_call_value)
    )
    if not is_retrieval_tool:
        return None
    
    # 获取第一个observation（应该在第一个function_call之后）
    first_obs_index = None
    for obs_idx in observations:
        if obs_idx > func_calls[0]:
            first_obs_index = obs_idx
            break
    
    if first_obs_index is None:
        return None
    
    first_obs_value = conversations[first_obs_index].get("value", "")
    
    # 获取第二个function_call（应该在第一个observation之后）
    second_func_call_index = None
    for func_idx in func_calls:
        if func_idx > first_obs_index:
            second_func_call_index = func_idx
            break
    
    if second_func_call_index is None:
        return None
    
    # 提取第二个function_call中的工具名称
    second_func_call_value = conversations[second_func_call_index].get("value", "")
    tool_name = None
    
    try:
        # 尝试解析JSON格式的function_call
        if second_func_call_value.startswith("{"):
            func_call_data = json.loads(second_func_call_value)
            tool_name = func_call_data.get("name")
    except (json.JSONDecodeError, KeyError):
        # 如果不是JSON格式，尝试用正则表达式提取
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', second_func_call_value)
        if name_match:
            tool_name = name_match.group(1)
    
    if not tool_name:
        return None
    
    return (first_obs_index, first_obs_value, second_func_call_index, tool_name)


def check_tool_in_observation(tool_name: str, observation_value: str) -> bool:
    """
    检查工具名称是否在observation的value中出现（关键词检索）
    """
    return tool_name in observation_value


async def call_qwen_api(session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> str:
    """异步调用Qwen API进行改写 - 使用greedy decoding（temperature=0）"""
    headers = {
        "Content-Type": "application/json"
    }
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    
    # 构建messages - 在系统提示中强调格式要求
    system_prompt = """你是一个专业的数据压缩专家。

重要：你必须严格按照以下XML标签格式返回结果：
<observation>
[这里是压缩后的observation内容]
</observation>

<gpt>
[这里是对应调整后的gpt回答]
</gpt>

不要添加任何其他内容，只返回这两个标签内的内容。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # 使用greedy decoding（temperature=0）确保最低负载和最高精度
    # 增加max_tokens确保能生成完整的标签
    data = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 3000,  # 限制最大生成长度
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    
    prompt_tokens = count_tokens(prompt)
    logger.info(f"准备调用Qwen API，prompt token数: {prompt_tokens}")
    logger.debug(f"Qwen API URL: {QWEN_API_URL}")
    logger.debug(f"Prompt预览: {prompt[:500]}...")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"第 {attempt + 1}/{max_retries} 次尝试调用Qwen API...")
            logger.info(f"正在发送请求到 {QWEN_API_URL}...")
            start_time = time.time()
            
            # 发送请求
            logger.info("✓ 请求已发送，等待Qwen模型生成响应...")
            logger.info("（使用greedy decoding，通常需要30-120秒，最长300秒超时）")
            
            # 创建一个任务来显示等待进度
            async def show_progress():
                """显示等待进度"""
                wait_time = 0
                while True:
                    await asyncio.sleep(10)
                    wait_time += 10
                    logger.info(f"⏳ 已等待 {wait_time} 秒...")
            
            progress_task = asyncio.create_task(show_progress())
            
            try:
                async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    # 取消进度显示任务
                    progress_task.cancel()
                    
                    elapsed = time.time() - start_time
                    logger.info(f"✓ 收到响应！耗时: {elapsed:.2f}秒, 状态码: {response.status}")
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Qwen API返回错误状态码: {response.status}")
                        logger.error(f"响应内容: {error_text[:1000]}")
                        raise Exception(f"API调用失败，状态码: {response.status}")
                    
                    result = await response.json()
                    logger.debug(f"Qwen API原始响应结构: {list(result.keys())}")
                    
                    # 提取文本内容
                    content = result['choices'][0]['message']['content']
                    
                    # 去除可能的thinking块
                    try:
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
                    except Exception:
                        pass
                    
                    content = content.strip()
                    result_tokens = count_tokens(content)
                    logger.info(f"Qwen API调用成功，返回token数: {result_tokens}")
                    logger.debug(f"返回文本预览: {content[:500]}...")
                    
                    return content
            except asyncio.CancelledError:
                # 进度任务被取消，这是正常的
                raise
            finally:
                # 确保取消进度任务
                if not progress_task.done():
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
            
        except asyncio.TimeoutError as e:
            # 取消进度任务
            if 'progress_task' in locals() and not progress_task.done():
                progress_task.cancel()
            logger.error(f"Qwen API请求超时 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # 增加等待时间，让服务器有更多时间恢复
                wait_time = 10 * (attempt + 1)  # 10秒、20秒、30秒
                logger.warning(f"服务器可能负载过高，等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Qwen API调用最终失败：超时（已重试3次，每次300秒）")
                raise
                
        except aiohttp.ClientError as e:
            logger.error(f"Qwen API请求异常 (尝试 {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Qwen API调用最终失败：请求异常")
                raise
                
        except (KeyError, IndexError) as e:
            logger.error(f"Qwen API响应解析失败 (尝试 {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
            logger.error(f"原始响应: {result if 'result' in locals() else 'N/A'}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Qwen API调用最终失败：响应解析错误")
                raise
                
        except Exception as e:
            logger.error(f"Qwen API调用未知错误 (尝试 {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Qwen API调用最终失败：未知错误")
                raise
    
    return ""


def truncate_observation(observation: str, max_tokens: int = 2000) -> str:
    """
    智能截断observation到指定token数
    特别处理图表数据中的长数组
    """
    obs_tokens = count_tokens(observation)
    if obs_tokens <= max_tokens:
        return observation
    
    logger.info(f"Observation token数 {obs_tokens} 超过限制 {max_tokens}，开始智能截断")
    
    # 尝试解析为JSON并简化图表数据
    try:
        obs_data = json.loads(observation)
        
        # 递归简化数组
        def simplify_arrays(obj, max_items=8):
            if isinstance(obj, list):
                # 如果是长数组（>15项），只保留前max_items项
                if len(obj) > 15:
                    logger.info(f"简化数组：从 {len(obj)} 项减少到 {max_items} 项")
                    return [simplify_arrays(item, max_items) for item in obj[:max_items]]
                else:
                    return [simplify_arrays(item, max_items) for item in obj]
            elif isinstance(obj, dict):
                return {k: simplify_arrays(v, max_items) for k, v in obj.items()}
            else:
                return obj
        
        simplified_data = simplify_arrays(obs_data, max_items=8)
        simplified_str = json.dumps(simplified_data, ensure_ascii=False, indent=2)
        simplified_tokens = count_tokens(simplified_str)
        
        if simplified_tokens <= max_tokens:
            logger.info(f"智能简化成功：从 {obs_tokens} tokens 减少到 {simplified_tokens} tokens")
            return simplified_str
        else:
            logger.info(f"智能简化后仍超标 ({simplified_tokens} tokens)，继续按字符截断")
    
    except json.JSONDecodeError:
        logger.info("非JSON格式，使用字符截断")
    except Exception as e:
        logger.warning(f"智能简化失败 ({e})，使用字符截断")
    
    # 按字符截断
    estimated_chars = max_tokens * 4
    truncated = observation[:estimated_chars]
    
    # 微调：如果还是超了，继续减少
    while count_tokens(truncated) > max_tokens and len(truncated) > 100:
        truncated = truncated[:int(len(truncated) * 0.9)]
    
    actual_tokens = count_tokens(truncated)
    logger.info(f"截断完成，从 {obs_tokens} tokens 截断到 {actual_tokens} tokens")
    return truncated


async def rewrite_observation_and_gpt(session: aiohttp.ClientSession, observation: str, gpt_response: str) -> tuple:
    """
    使用Qwen API改写observation和gpt回答
    主要目的是压缩过长的列表数据（如订单列表）
    """
    # 先截断observation（如果太长）
    original_obs_tokens = count_tokens(observation)
    truncated_observation = truncate_observation(observation, max_tokens=2000)
    truncated_obs_tokens = count_tokens(truncated_observation)
    
    # 同时截断gpt_response（如果太长）
    original_gpt_tokens = count_tokens(gpt_response)
    if original_gpt_tokens > 2000:
        truncated_gpt = gpt_response[:6000]  # 保守截断
        logger.info(f"GPT回答过长，从 {original_gpt_tokens} tokens 截断到约 {count_tokens(truncated_gpt)} tokens")
    else:
        truncated_gpt = gpt_response
    
    prompt = f"""任务：将observation数据从{original_obs_tokens} tokens压缩到约1000-1500 tokens。

【数据样本】
下面是截取的前2000 tokens作为参考：

observation:
{truncated_observation}

gpt回答:
{truncated_gpt}

【压缩规则】
1. observation压缩：
   - 如果是订单/产品列表：只保留8-10条代表性数据
   - 如果包含图表数据（chart_data）：将数组简化为5-8个代表性数据点（不要生成完整的几十个数据点！）
   - 如果有长数组（如categories: [日期1, 日期2, ..., 日期68]），只保留前5-8个
   - 保持JSON格式正确，确保所有括号闭合
2. gpt回答压缩：
   - 必须是自然语言总结，不要输出纯JSON格式
   - 调整数量与压缩后observation匹配（如原78条改为8条）
   - 保持markdown格式和语言风格
3. 重要：生成长度控制在2000 tokens以内，不要生成过长的数组！

【特别注意】
- 如果observation中有chart_data的数组（如categories: [很多日期], data: [很多数字]），必须大幅简化！
- 例如：categories有68个日期→只保留5-8个代表性日期
- 例如：data有68个数字→只保留对应的5-8个数字
- 数组简化示例：[1,2,3,4,5,...,68] → [1,2,3,4,5,6,7,8]

【输出格式 - 必须严格遵循】
<observation>
[压缩后的JSON，数组只保留5-8项，总长度2000 tokens以内]
</observation>

<gpt>
[自然语言总结，不是JSON代码块]
</gpt>

现在请直接返回压缩结果，不要添加任何其他解释："""
    
    logger.info(f"调用Qwen API改写，原始observation token: {original_obs_tokens} -> 截断后: {truncated_obs_tokens}, gpt token: {original_gpt_tokens}")
    logger.info(f"使用greedy decoding (temperature=0) 确保输出精度和最低负载")
    
    result = await call_qwen_api(session, prompt)
    
    # 解析返回结果 - 使用更健壮的匹配
    # 尝试贪婪匹配（.*?改为[\s\S]*?）以处理长内容
    obs_match = re.search(r'<observation>\s*([\s\S]*?)\s*</observation>', result, re.DOTALL)
    gpt_match = re.search(r'<gpt>\s*([\s\S]*?)\s*</gpt>', result, re.DOTALL)
    
    # 如果第一种匹配失败，尝试更宽松的匹配
    if not obs_match:
        logger.warning("标准匹配失败，尝试宽松匹配...")
        # 查找<observation>到文件末尾或</gpt>之前的内容
        obs_match_loose = re.search(r'<observation>([\s\S]*?)(?:</observation>|<gpt>)', result)
        if obs_match_loose:
            obs_match = obs_match_loose
    
    if not gpt_match:
        # 查找<gpt>到文件末尾的内容
        gpt_match_loose = re.search(r'<gpt>([\s\S]*?)(?:</gpt>|$)', result)
        if gpt_match_loose:
            gpt_match = gpt_match_loose
    
    if obs_match and gpt_match:
        new_observation = obs_match.group(1).strip()
        new_gpt = gpt_match.group(1).strip()
        
        new_obs_tokens = count_tokens(new_observation)
        new_gpt_tokens = count_tokens(new_gpt)
        
        logger.info(f"✓ 解析成功！新observation token: {new_obs_tokens}, 新gpt token: {new_gpt_tokens}")
        logger.info(f"\n{'='*60}")
        logger.info(f"改写后的observation（前2000字符）:")
        logger.info(f"{new_observation}")
        logger.info(f"\n改写后的gpt（完整）:")
        logger.info(f"{new_gpt}")
        logger.info(f"{'='*60}\n")
        
        return new_observation, new_gpt
    else:
        logger.error("✗ 解析失败！")
        logger.error(f"  - observation匹配: {'成功' if obs_match else '失败'}")
        logger.error(f"  - gpt匹配: {'成功' if gpt_match else '失败'}")
        logger.error(f"\n返回内容长度: {len(result)} 字符")
        logger.error(f"返回内容预览（前1500字符）:\n{result[:1500]}...")
        logger.error(f"\n返回内容结尾（后500字符）:\n...{result[-500:]}")
        
        # 标记为失败，返回空值
        logger.error("由于解析失败，返回空值（将触发删除）")
        return "", ""


async def process_data_item(session: aiohttp.ClientSession, data: Dict[str, Any], index: int) -> tuple:
    """
    处理单条数据
    返回：(status, new_data)
    status: 'skip'=跳过不需要改写, 'success'=改写成功, 'failed'=改写失败需删除
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始处理第 {index + 1} 条数据")
    logger.info(f"{'='*60}")
    
    total_tokens = calculate_data_tokens(data)
    logger.info(f"总token数: {total_tokens}")
    
    # 如果token < 7000，不修改
    if total_tokens < 7000:
        logger.info(f"✓ Token数 {total_tokens} < 7000，无需改写")
        return 'skip', data
    
    logger.warning(f"⚠ Token数 {total_tokens} >= 7000，需要检查是否改写")
    
    # 检查两轮function_call结构：检测第二个function_call中的工具是否在第一个observation中
    conversations = data.get("conversations", [])
    two_round_structure = check_two_round_function_call_structure(conversations)
    
    if two_round_structure:
        first_obs_index, first_obs_value, second_func_call_index, tool_name = two_round_structure
        logger.info(f"检测到两轮function_call结构：")
        logger.info(f"  - 第一个observation索引: {first_obs_index}")
        logger.info(f"  - 第二个function_call索引: {second_func_call_index}")
        logger.info(f"  - 调用的工具名称: {tool_name}")
        
        # 检查工具名称是否在第一个observation中出现
        tool_found = check_tool_in_observation(tool_name, first_obs_value)
        if tool_found:
            logger.info(f"✓ 工具 '{tool_name}' 在第一个observation中找到")
        else:
            logger.warning(f"⚠ 工具 '{tool_name}' 未在第一个observation中找到（可能存在问题）")
    else:
        logger.info("未检测到两轮function_call结构，跳过工具检查")
    
    # 检查最后一个observation
    last_obs = get_last_observation(conversations)
    if not last_obs:
        logger.info("✓ 未找到observation，跳过")
        return 'skip', data
    
    obs_index, obs_value, obs_tokens = last_obs
    logger.info(f"最后一个observation位于索引 {obs_index}，token数: {obs_tokens}")
    
    # 如果最后一个observation的token <= 3500，不修改
    if obs_tokens <= 3500:
        logger.info(f"✓ 最后一个observation token数 {obs_tokens} <= 3500，无需改写")
        return 'skip', data
    
    logger.warning(f"⚠ 最后一个observation token数 {obs_tokens} > 3500，需要改写")
    
    # 需要改写：找到对应的gpt回答
    gpt_index = None
    gpt_value = None
    
    # 找到observation后面的gpt回答
    for i in range(obs_index + 1, len(conversations)):
        if conversations[i].get("from") == "gpt":
            gpt_index = i
            gpt_value = conversations[i]["value"]
            break
    
    if gpt_index is None:
        logger.warning("✗ 未找到对应的gpt回答，标记为失败（删除）")
        return 'failed', data
    
    logger.info(f"找到对应的gpt回答位于索引 {gpt_index}")
    logger.info(f"开始调用Qwen API进行改写...")
    
    try:
        # 调用Qwen API改写
        new_observation, new_gpt = await rewrite_observation_and_gpt(session, obs_value, gpt_value)
        
        if not new_observation or not new_gpt:
            logger.error("✗ 改写结果为空，标记为失败（删除）")
            return 'failed', data
        
        # 创建新的数据副本
        new_data = json.loads(json.dumps(data))
        new_data["conversations"][obs_index]["value"] = new_observation
        new_data["conversations"][gpt_index]["value"] = new_gpt
        
        # 计算新的token数
        new_total_tokens = calculate_data_tokens(new_data)
        new_obs_tokens = count_tokens(new_observation)
        reduction = total_tokens - new_total_tokens
        reduction_pct = (reduction / total_tokens) * 100 if total_tokens > 0 else 0
        
        logger.info(f"✓ 改写成功！")
        logger.info(f"  - 总token: {total_tokens} -> {new_total_tokens} (减少 {reduction} tokens, {reduction_pct:.1f}%)")
        logger.info(f"  - observation token: {obs_tokens} -> {new_obs_tokens}")
        
        # 添加延迟，让服务器有时间清理资源，避免队列堆积
        logger.info("等待5秒，让服务器清理资源...")
        await asyncio.sleep(5)
        
        return 'success', new_data
        
    except Exception as e:
        logger.error(f"✗ 改写失败: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        logger.warning("标记为失败（删除）")
        return 'failed', data


async def check_server_health(session: aiohttp.ClientSession) -> bool:
    """检查服务器健康状态"""
    try:
        logger.info("检查服务器健康状态...")
        models_url = f"{VLLM_BASE_URL.rstrip('/')}/v1/models"
        async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                logger.info("✓ 服务器健康检查通过")
                return True
            else:
                logger.warning(f"⚠ 服务器返回状态码: {response.status}")
                return False
    except Exception as e:
        logger.error(f"✗ 服务器健康检查失败: {e}")
        return False


async def process_json_file(input_file: str, output_file: str, start_idx: int = 0, end_idx: Optional[int] = None):
    """
    异步处理整个JSON文件
    """
    logger.info(f"开始处理文件: {input_file}")
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    total_count = len(data_list)
    logger.info(f"总共 {total_count} 条数据")
    
    # 确定处理范围
    if end_idx is None:
        end_idx = total_count
    else:
        end_idx = min(end_idx, total_count)
    
    logger.info(f"处理范围: {start_idx} 到 {end_idx - 1}")
    
    # 创建aiohttp会话
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 启动前先检查服务器健康状态
        await check_server_health(session)
        
        # 处理数据
        success_count = 0    # 改写成功
        skipped_count = 0    # 跳过（不需要改写）
        deleted_count = 0    # 删除（改写失败）
        error_count = 0      # 处理异常
        result_data = []
        consecutive_failures = 0  # 连续失败计数
        
        start_time = time.time()
        
        for i in range(start_idx, end_idx):
            data = data_list[i]
            
            try:
                status, new_data = await process_data_item(session, data, i)
                
                if status == 'success':
                    success_count += 1
                    result_data.append(new_data)
                    consecutive_failures = 0  # 重置连续失败计数
                elif status == 'skip':
                    skipped_count += 1
                    result_data.append(new_data)
                    consecutive_failures = 0  # 重置连续失败计数
                elif status == 'failed':
                    deleted_count += 1
                    consecutive_failures += 1
                    # 不添加到result_data，相当于删除
                    logger.warning(f"✗ 第 {i + 1} 条数据改写失败，已删除 (连续失败: {consecutive_failures})")
                    
                    # 如果连续失败3次，检查服务器健康并等待
                    if consecutive_failures >= 3:
                        logger.warning(f"⚠⚠⚠ 连续失败 {consecutive_failures} 次，检查服务器状态...")
                        healthy = await check_server_health(session)
                        if healthy:
                            logger.info("服务器正常，等待30秒后继续...")
                            await asyncio.sleep(30)
                        else:
                            logger.error("服务器异常，等待60秒后继续...")
                            await asyncio.sleep(60)
                        consecutive_failures = 0  # 重置计数
                
            except Exception as e:
                logger.error(f"处理第 {i + 1} 条数据时发生严重错误: {e}")
                error_count += 1
                deleted_count += 1
                consecutive_failures += 1
                # 异常也不添加，相当于删除
                logger.warning(f"✗ 第 {i + 1} 条数据处理异常，已删除 (连续失败: {consecutive_failures})")
            
            # 计算进度和预估时间
            processed = i - start_idx + 1
            total = end_idx - start_idx
            progress_pct = (processed / total) * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            remaining = (total - processed) * avg_time
            
            # 每处理5条数据显示一次进度
            if processed % 5 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"进度: {processed}/{total} ({progress_pct:.1f}%)")
                logger.info(f"改写成功: {success_count} | 跳过: {skipped_count} | 已删除: {deleted_count} | 异常: {error_count}")
                logger.info(f"当前输出数据量: {len(result_data)} 条")
                logger.info(f"已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟")
                logger.info(f"{'='*60}\n")
        
        elapsed_total = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"处理完成！总耗时: {elapsed_total/60:.1f}分钟")
        logger.info(f"总计处理: {end_idx - start_idx} 条数据")
        logger.info(f"  - 改写成功: {success_count} 条")
        logger.info(f"  - 跳过（无需改写）: {skipped_count} 条")
        logger.info(f"  - 已删除（改写失败）: {deleted_count} 条")
        logger.info(f"  - 处理异常（已删除）: {error_count} 条")
        logger.info(f"  - 最终保留: {len(result_data)} 条数据")
        logger.info(f"{'='*60}\n")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="训练数据改写脚本")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出JSON文件路径")
    parser.add_argument("--start", "-s", type=int, default=0, help="开始索引（默认0）")
    parser.add_argument("--end", "-e", type=int, default=None, help="结束索引（默认处理到最后）")
    parser.add_argument("--log", "-l", type=str, default="rewrite_long_data.log", help="日志文件路径")
    parser.add_argument("--debug", "-d", action="store_true", help="启用DEBUG级别日志")
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(args.log, rotation="100 MB", level=log_level)
    
    logger.info(f"{'='*60}")
    logger.info(f"训练数据改写脚本启动")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"处理范围: {args.start} - {args.end if args.end else '最后'}")
    logger.info(f"日志文件: {args.log}")
    logger.info(f"日志级别: {log_level}")
    logger.info(f"{'='*60}\n")
    
    # 处理文件（异步）
    try:
        asyncio.run(process_json_file(args.input, args.output, args.start, args.end))
        logger.info("\n脚本执行成功完成！")
    except KeyboardInterrupt:
        logger.warning("\n用户中断执行")
    except Exception as e:
        logger.error(f"\n脚本执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
    '''
    # 激活环境
    conda activate my_qwen

    # 安装依赖（如果没有）
    pip install tiktoken loguru requests

    # 处理整个文件
    python /home/ziqiang/LLaMA-Factory/data/function_call_data/rewrite_long_data.py \
    --input  /home/ziqiang/LLaMA-Factory/data/dataset/11_15/rank_orders_data.json \
    --output /home/ziqiang/LLaMA-Factory/data/dataset/11_15/rank_orders_data.json

    # 处理指定范围（例如前100条）
    python /home/ziqiang/LLaMA-Factory/data/function_call_data/rewrite_long_data.py \
    --input /home/ziqiang/LLaMA-Factory/data/function_call_data/10.12_train_data_top5.json \
    --output /home/ziqiang/LLaMA-Factory/data/function_call_data/10.12_train_data_top5_rewritten.json \
    --start 0 --end 100
    
    '''
