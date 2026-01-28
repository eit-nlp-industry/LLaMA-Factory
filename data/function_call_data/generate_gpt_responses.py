#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为训练数据生成GPT回复
根据每条数据中最后的observation内容，调用本地大模型生成对应的gpt回复
"""

import json
import os
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from loguru import logger
import aiohttp

# Qwen API配置（vLLM兼容）
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://125.122.38.32:8027")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
QWEN_MODEL_NAME = "/data/models/Qwen3-8B"
QWEN_API_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"


def get_last_observation(conversations: List[Dict]) -> Optional[tuple]:
    """
    获取最后一个observation及其索引
    返回：(index, observation_value) 或 None
    """
    for i in range(len(conversations) - 1, -1, -1):
        if conversations[i].get("from") == "observation":
            obs_value = conversations[i]["value"]
            return (i, obs_value)
    return None


def get_human_query(conversations: List[Dict]) -> str:
    """
    获取用户的原始问题
    """
    for conv in conversations:
        if conv.get("from") == "human":
            return conv["value"]
    return ""


async def call_qwen_api(session: aiohttp.ClientSession, prompt: str, max_retries: int = 3) -> str:
    """异步调用Qwen API生成gpt回复"""
    headers = {
        "Content-Type": "application/json"
    }
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    
    # 构建messages
    system_prompt = """你是一个智能助手，需要根据用户的问题和工具返回的observation数据，生成简洁、准确的自然语言回复。

要求：
1. 回复必须是自然语言，不要输出JSON格式或代码块
2. 从observation中提取关键信息进行总结
3. 回复要简洁明了，直接回答用户问题
4. 如果observation包含数据统计，要提炼关键数字和结论
5. 保持专业、友好的语气
6. 只输出最终回复内容，不要添加任何标签或解释"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # 使用较低的temperature确保稳定性
    data = {
        "model": QWEN_MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 1000,
        "stream": False,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
    
    logger.info(f"准备调用Qwen API生成回复")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"第 {attempt + 1}/{max_retries} 次尝试调用Qwen API...")
            start_time = time.time()
            
            async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                elapsed = time.time() - start_time
                logger.info(f"✓ 收到响应！耗时: {elapsed:.2f}秒, 状态码: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Qwen API返回错误状态码: {response.status}")
                    logger.error(f"响应内容: {error_text[:1000]}")
                    raise Exception(f"API调用失败，状态码: {response.status}")
                
                result = await response.json()
                
                # 提取文本内容
                content = result['choices'][0]['message']['content']
                content = content.strip()
                
                logger.info(f"Qwen API调用成功")
                logger.debug(f"生成的回复: {content[:200]}...")
                
                return content
            
        except asyncio.TimeoutError as e:
            logger.error(f"Qwen API请求超时 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                logger.warning(f"等待 {wait_time} 秒后重试...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Qwen API调用最终失败：超时")
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


async def generate_gpt_response(session: aiohttp.ClientSession, human_query: str, observation: str) -> str:
    """
    根据用户问题和observation生成gpt回复
    """
    prompt = f"""用户问题：
{human_query}

工具返回的observation数据：
{observation}

请根据上述信息，生成一个自然、简洁的回复来回答用户的问题。只输出回复内容，不要添加任何额外说明。"""
    
    logger.info(f"生成GPT回复...")
    logger.debug(f"用户问题: {human_query[:100]}...")
    logger.debug(f"Observation长度: {len(observation)} 字符")
    
    try:
        gpt_response = await call_qwen_api(session, prompt)
        
        if not gpt_response:
            logger.error("✗ 生成的回复为空")
            return ""
        
        logger.info(f"✓ 成功生成回复，长度: {len(gpt_response)} 字符")
        return gpt_response
        
    except Exception as e:
        logger.error(f"✗ 生成回复失败: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
        return ""


async def process_data_item(session: aiohttp.ClientSession, data: Dict[str, Any], index: int) -> tuple:
    """
    处理单条数据
    返回：(status, new_data)
    status: 'success'=成功生成, 'skip'=跳过（已有gpt回复）, 'failed'=生成失败
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始处理第 {index + 1} 条数据")
    logger.info(f"{'='*60}")
    
    conversations = data.get("conversations", [])
    
    # 检查最后一条是否已经是gpt回复
    if conversations and conversations[-1].get("from") == "gpt":
        logger.info(f"✓ 该条数据已有GPT回复，跳过")
        return 'skip', data
    
    # 获取最后一个observation
    last_obs = get_last_observation(conversations)
    if not last_obs:
        logger.warning("✗ 未找到observation，跳过")
        return 'skip', data
    
    obs_index, obs_value = last_obs
    logger.info(f"找到observation位于索引 {obs_index}")
    
    # 获取用户问题
    human_query = get_human_query(conversations)
    if not human_query:
        logger.warning("✗ 未找到用户问题，跳过")
        return 'skip', data
    
    logger.info(f"用户问题: {human_query[:100]}...")
    
    try:
        # 生成gpt回复
        gpt_response = await generate_gpt_response(session, human_query, obs_value)
        
        if not gpt_response:
            logger.error("✗ 生成失败，回复为空")
            return 'failed', data
        
        # 创建新的数据副本并添加gpt回复
        new_data = json.loads(json.dumps(data))
        new_data["conversations"].append({
            "from": "gpt",
            "value": gpt_response
        })
        
        logger.info(f"✓ 成功生成GPT回复")
        logger.info(f"  回复内容: {gpt_response[:200]}...")
        
        # 添加延迟，避免请求过快
        logger.info("等待2秒...")
        await asyncio.sleep(2)
        
        return 'success', new_data
        
    except Exception as e:
        logger.error(f"✗ 处理失败: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"错误堆栈:\n{traceback.format_exc()}")
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
    timeout = aiohttp.ClientTimeout(total=120)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 启动前先检查服务器健康状态
        await check_server_health(session)
        
        # 处理数据
        success_count = 0    # 成功生成
        skipped_count = 0    # 跳过（已有或无需生成）
        failed_count = 0     # 生成失败
        result_data = []
        
        start_time = time.time()
        
        for i in range(start_idx, end_idx):
            data = data_list[i]
            
            try:
                status, new_data = await process_data_item(session, data, i)
                
                if status == 'success':
                    success_count += 1
                    result_data.append(new_data)
                elif status == 'skip':
                    skipped_count += 1
                    result_data.append(new_data)
                elif status == 'failed':
                    failed_count += 1
                    result_data.append(data)  # 保留原始数据
                    logger.warning(f"✗ 第 {i + 1} 条数据生成失败，保留原始数据")
                
            except Exception as e:
                logger.error(f"处理第 {i + 1} 条数据时发生严重错误: {e}")
                failed_count += 1
                result_data.append(data)  # 保留原始数据
            
            # 计算进度和预估时间
            processed = i - start_idx + 1
            total = end_idx - start_idx
            progress_pct = (processed / total) * 100
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            remaining = (total - processed) * avg_time
            
            # 每处理10条数据显示一次进度
            if processed % 10 == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"进度: {processed}/{total} ({progress_pct:.1f}%)")
                logger.info(f"成功生成: {success_count} | 跳过: {skipped_count} | 失败: {failed_count}")
                logger.info(f"已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟")
                logger.info(f"{'='*60}\n")
        
        elapsed_total = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"处理完成！总耗时: {elapsed_total/60:.1f}分钟")
        logger.info(f"总计处理: {end_idx - start_idx} 条数据")
        logger.info(f"  - 成功生成: {success_count} 条")
        logger.info(f"  - 跳过: {skipped_count} 条")
        logger.info(f"  - 失败: {failed_count} 条")
        logger.info(f"  - 最终输出: {len(result_data)} 条数据")
        logger.info(f"{'='*60}\n")
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="为训练数据生成GPT回复")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出JSON文件路径")
    parser.add_argument("--start", "-s", type=int, default=0, help="开始索引（默认0）")
    parser.add_argument("--end", "-e", type=int, default=None, help="结束索引（默认处理到最后）")
    parser.add_argument("--log", "-l", type=str, default="generate_gpt_responses.log", help="日志文件路径")
    parser.add_argument("--debug", "-d", action="store_true", help="启用DEBUG级别日志")
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(args.log, rotation="100 MB", level=log_level)
    
    logger.info(f"{'='*60}")
    logger.info(f"GPT回复生成脚本启动")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"处理范围: {args.start} - {args.end if args.end else '最后'}")
    logger.info(f"日志文件: {args.log}")
    logger.info(f"日志级别: {log_level}")
    logger.info(f"API地址: {QWEN_API_URL}")
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

