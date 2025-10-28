#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据评估脚本
根据9.17_evaluate_data_top5_final.json的数据结构，将conversations分成source-target pairs，
使用LLM生成预测并评估工具调用和文本生成的质量
"""

import json
import asyncio
import re
import sys
import os
import time
import requests
import argparse
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from pathlib import Path
from collections import defaultdict
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import signal

# 递归保留浮点到小数点后三位
def _round_floats(obj: Any, ndigits: int = 3) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj

# Gemini API Key
GEMINI_API_KEY = "AIzaSyDikJjktaSUq3sJCAHUIu7JmMEgP1DeHSI"

# vLLM/OpenAI 兼容 Chat Completions 配置（可通过环境变量覆盖）
# VLLM_BASE_URL 例如: http://127.0.0.1:8000
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://125.122.38.32:8021")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")

QWEN_MODEL_NAME = "my_lora"
#QWEN_MODEL_NAME = "/data/models/Qwen3-8B"

# 统一的 Chat Completions 端点（vLLM/OpenAI 兼容）
QWEN_API_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"
# Retrieval Tool API 配置
RETRIEVAL_ENDPOINT = "http://125.122.38.32:8024/retrieval_tool"
RETRIEVAL_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

# 临时开关：跳过调用 8024 检索服务与相关 recall 指标统计
DISABLE_RECALL = str(os.getenv("EVAL_DISABLE_RECALL", "0")).lower() in ("1", "true", "yes")

# 并发控制配置
MAX_CONCURRENT_CONVERSATIONS = int(os.getenv("MAX_CONCURRENT_CONVERSATIONS", "5"))  # 最大并发对话数
MAX_CONCURRENT_PAIRS = int(os.getenv("MAX_CONCURRENT_PAIRS", "10"))  # 最大并发pair数
MAX_CONCURRENT_API_CALLS = int(os.getenv("MAX_CONCURRENT_API_CALLS", "20"))  # 最大并发API调用数

@dataclass
class EvaluationPair:
    """评估对结构"""
    pair_id: int
    source: str  # system + tools + user/observation
    target: str  # 期望的输出
    pair_type: str  # 'tool_call' 或 'text_generation'
    conversation_id: int  # 新增：对话ID

@dataclass
class EvaluationResult:
    """评估结果结构"""
    conversation_id: int  # 新增：对话ID
    pair_id: int
    pair_type: str
    source: str
    target: str
    predict: str
    score: float
    tool_name_score: float
    recall: Optional[int] = None  # 新增：recall指标 (0或1)
    recall_details: Optional[Dict[str, Any]] = None  # 新增：recall详细信息
    details: Dict[str, Any] = None

@dataclass
class RealTimeMetrics:
    """实时指标结构"""
    total_conversations: int = 0
    total_pairs: int = 0
    
    # 按pair_id分组的指标
    pair1: Dict[str, float] = None  # pair1指标（不涉及recall）
    
    pair2: Dict[str, float] = None
    pair2_consider_recall: Dict[str, float] = None      # 考虑recall的pair2指标（仅在recall=1条件下计算）
    
    # 文本生成指标
    pair3: Dict[str, float] = None
    
    # recall指标
    recall_metrics: Dict[str, Any] = None
    
    # 总体指标
    overall_current_logic: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pair1 is None:
            self.pair1 = {"total": 0, "accuracy": 0.0, "precision@1": 0.0}
        
        if self.pair2 is None:
            self.pair2 = {"total": 0, "accuracy": 0.0, "precision@1": 0.0}
        if self.pair2_consider_recall is None:
            self.pair2_consider_recall = {"total": 0, "accuracy": 0.0, "precision@1": 0.0}
        
        if self.pair3 is None:
            self.pair3 = {"total": 0, "answer_score": 0.0}
        
        if self.recall_metrics is None:
            self.recall_metrics = {"total_pairs": 0, "recall@5_1": 0, "recall@5_0": 0, "recall_rate": 0.0}
        
        if self.overall_current_logic is None:
            self.overall_current_logic = {"total": 0, "accuracy": 0.0, "precision@1": 0.0, "answer_score": 0.0}

class DataProcessor:
    """数据处理模块：将conversations分割成source-target pairs"""
    
    def __init__(self):
        logger.info("初始化数据处理模块")
    
    def parse_conversations(self, conversation_data: Dict, conversation_id: int) -> List[EvaluationPair]:
        """
        解析conversations数据，分割成pairs
        - Pair 1: system+tools+user -> function_call
        - Pair 2: system+tools+user+observation -> function_call  
        - Pair 3: system+tools+user+observation -> gpt
        """
        conversations = conversation_data["conversations"]
        system_prompt = conversation_data["system"]
        tools = conversation_data.get("tools", "[]")
        
        pairs = []
        pair_id = 1
        
        # 提取原始用户query
        original_query = ""
        for msg in conversations:
            if msg["from"] == "human":
                original_query = msg["value"]
                break
        
        # 使用原始的system prompt，并注入tools内容（兼容空标签/已有内容）
        try:
            tools_str = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
        except Exception:
            tools_str = str(tools)

        if '<tools>' in system_prompt and '</tools>' in system_prompt:
            # 与训练对齐：中文段落中的 <tools></tools> 保持为空，避免与英文段落重复
            try:
                base_system = re.sub(r'<tools>\s*[\s\S]*?</tools>', '<tools>\n</tools>', system_prompt)
            except Exception:
                base_system = system_prompt.replace('<tools>\n</tools>', '<tools>\n</tools>').replace('<tools></tools>', '<tools>\n</tools>')
        else:
            # 如果没有tools标签，直接使用原始system
            base_system = system_prompt

        # 追加英文模板与英文 <tools>（与训练模板对齐）
        try:
            parsed_tools = json.loads(tools) if isinstance(tools, str) else tools
        except Exception:
            parsed_tools = tools

        try:
            if isinstance(parsed_tools, list) and parsed_tools and isinstance(parsed_tools[0], dict):
                english_tools_obj = {"type": "function", "function": parsed_tools[0]}
                english_tools_str = json.dumps(english_tools_obj, ensure_ascii=False)
            else:
                english_tools_str = tools_str
        except Exception:
            english_tools_str = tools_str

        # 英文模板固定化，逐字对齐训练日志
        english_tail = (
            "\n\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{english_tools_str}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call>"
        )

        base_system = f"{base_system}{english_tail}"
        
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            
            if msg["from"] == "human":
                # Pair 1: system + tools + user -> function_call
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "function_call":
                    # system 中不再追加“只输出一个<tool_call>...”提示
                    source = f"{base_system}\n\nUser: {msg['value']}"
                    target = conversations[i + 1]["value"]
                    pairs.append(EvaluationPair(
                        pair_id=pair_id,
                        source=source,
                        target=target,
                        pair_type="tool_call",
                        conversation_id=conversation_id
                    ))
                    pair_id += 1
                    i += 2
                else:
                    i += 1
            
            elif msg["from"] == "observation":
                # 查找下一个非observation的消息
                if i + 1 < len(conversations):
                    next_msg = conversations[i + 1]
                    if next_msg["from"] == "function_call":
                        # Pair 2: system + <tool_response>...</tool_response> -> function_call（对齐训练日志风格）
                        tool_resp_block = (
                            f"<tool_response>\n"
                            f"用户查询: {original_query}\n\n"
                            f"工具返回结果: {msg['value']}\n"
                            f"</tool_response>"
                        )
                        source = f"{base_system}\n\n{tool_resp_block}"
                        target = next_msg["value"]
                        pairs.append(EvaluationPair(
                            pair_id=pair_id,
                            source=source,
                            target=target,
                            pair_type="tool_call",
                            conversation_id=conversation_id
                        ))
                        pair_id += 1
                        i += 2
                    elif next_msg["from"] == "gpt":
                        # Pair 3: system + <tool_response>...</tool_response> -> gpt（对齐训练日志风格）
                        # 注释掉：不再评估文本生成部分
                        # tool_resp_block = (
                        #     f"<tool_response>\n"
                        #     f"用户查询: {original_query}\n\n"
                        #     f"工具返回结果: {msg['value']}\n"
                        #     f"</tool_response>"
                        # )
                        # source = f"{base_system}\n\n{tool_resp_block}"
                        # target = next_msg["value"]
                        # pairs.append(EvaluationPair(
                        #     pair_id=pair_id,
                        #     source=source,
                        #     target=target,
                        #     pair_type="text_generation",
                        #     conversation_id=conversation_id
                        # ))
                        # pair_id += 1
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        logger.info(f"成功解析出 {len(pairs)} 个评估对 (conversation_id: {conversation_id})")
        return pairs

class LLMPredictor:
    """LLM预测模块：根据source生成predict，使用Qwen API"""
    
    def __init__(self, model_type: str = "qwen3"):
        self.model_type = QWEN_MODEL_NAME  # 使用全局配置的模型名称
        self.max_retries = 5
        self.retry_delay = 10
        logger.info(f"初始化LLM预测模块，使用模型: {self.model_type}")
    
    async def call_qwen_api(self, session: aiohttp.ClientSession, prompt: List[Dict], temperature: float = 0.0, top_p: float = 1.0) -> str:
        """异步调用Qwen API生成预测"""
        headers = {
            "Content-Type": "application/json"
        }
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
        
        data = {
            "model": self.model_type,
            "messages": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        # 保险起见，去除可能残留的 <think> 块
                        try:
                            content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
                        except Exception:
                            pass
                        logger.debug(f"LLM 返回片段: {content[:400]}")
                        return content.strip()
                    else:
                        error_msg = f"API调用失败，状态码: {response.status}, 响应: {await response.text()}"
                        if attempt < self.max_retries - 1:
                            logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                            await asyncio.sleep(2 ** attempt)  # 指数退避：2^0, 2^1, 2^2秒
                        else:
                            raise Exception(error_msg)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error_msg = f"网络请求异常: {str(e)}"
                if attempt < self.max_retries - 1:
                    logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise Exception(error_msg)
        
        return ""
    
    async def predict(self, session: aiohttp.ClientSession, source: str, pair_type: str) -> str:
        """根据source生成预测：将用户内容放到 user 角色，system 仅保留指令与工具。"""
        try:
            system_content = source
            user_content = None

            # 规则1（优先）：如果包含 "\n\nUser: "，优先将其之后的原始问题作为 user（保证 Pair 1 正确）
            if "\n\nUser: " in source:
                parts = source.split("\n\nUser: ", 1)
                system_content = parts[0]
                user_content = parts[1]
                # 保持原始用户问题不变，不进行任何修改

            # 规则2（其次）：如果包含 <tool_response>，提取用户查询和工具返回结果作为 user 内容（适用于 Pair 2/3）
            if user_content is None:
                tool_resp_match = re.search(r'<tool_response>[\s\S]*?</tool_response>', source)
                if tool_resp_match:
                    tool_resp_content = tool_resp_match.group(0)
                    # 从tool_response中提取用户查询和工具返回结果
                    user_query_match = re.search(r'用户查询:\s*(.+?)(?:\n\n|$)', tool_resp_content)
                    tool_result_match = re.search(r'工具返回结果:\s*(.+?)(?:\n|$)', tool_resp_content, re.DOTALL)
                    
                    if user_query_match and tool_result_match:
                        user_query = user_query_match.group(1).strip()
                        tool_result = tool_result_match.group(1).strip()
                        user_content = f"用户问题：{user_query}\n\n工具返回结果：{tool_result}"
                    else:
                        # 如果无法解析，使用原始tool_response内容
                        user_content = tool_resp_content
                    
                    system_content = source.replace(tool_resp_content, "").strip()

            # 清理：确保 system 不包含任何残留的 "User: ..." 段落
            if "\n\nUser: " in system_content:
                system_content = system_content.split("\n\nUser: ", 1)[0].rstrip()

            # 默认用户内容兜底
            if user_content is None:
                user_content = ""

            # 为不同 pair 类型补充用户端约束指令
            if pair_type == "tool_call":
                if user_content.strip():
                    # 保持原始用户问题，只添加约束指令
                    user_content = f"{user_content}\n\n只输出一个<tool_call>，不要输出解释性文本或答案。"
                else:
                    user_content = "只输出一个<tool_call>，不要输出解释性文本或答案。"
            else:
                if not user_content.strip():
                    user_content = "请根据工具返回的结果生成最终回答。"

            prompt = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            logger.info(f"LLM prompt: {prompt}, user指令: {( 'tool_call' if pair_type=='tool_call' else 'text_generation')} ")
            # 调用Qwen API
            result = await self.call_qwen_api(session, prompt, temperature=0.0, top_p=1.0)
            logger.info(f"LLM 输出长度: {len(result)}，预览: {result[:5000]}")
            return result
        except Exception as e:
            logger.error(f"LLM预测失败: {e}")
            return ""

class RetrievalToolCaller:
    """检索工具调用模块"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 2
        logger.info("初始化检索工具调用模块")
    
    def extract_query_params(self, pair1_source: str) -> Dict[str, Any]:
        """从pair1的source中提取查询参数"""
        try:
            # 尝试从source中提取用户查询
            user_query = ""
            if "User: " in pair1_source:
                user_query = pair1_source.split("User: ")[1].strip()
            
            # 构建retrieval_tool的调用参数
            params = {
                "query": user_query,
                "source_filter": "toollist",
                "user_id": 136451106,  # 使用默认用户ID
                "top_k": 5
            }
            return params
        except Exception as e:
            logger.error(f"提取查询参数失败: {e}")
            return {}

    def _extract_tool_call_from_text(self, text: str) -> Dict[str, Any]:
        """从模型预测文本中提取工具调用对象（支持裸 JSON 或 <tool_call>{...}</tool_call>）"""
        try:
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            match = re.search(r'<tool_call>\s*({[\s\S]*?})\s*</tool_call>', text)
            if match:
                return json.loads(match.group(1))
            # 最后尝试整体解析
            return json.loads(text)
        except Exception:
            return {}

    def extract_query_params_from_pair1_predict(self, pair1_predict: str) -> Dict[str, Any]:
        """从 pair1 的预测结果中提取检索参数（使用 predict_call.arguments.query）"""
        try:
            call_obj = self._extract_tool_call_from_text(pair1_predict)
            arguments = call_obj.get("arguments", {}) if isinstance(call_obj, dict) else {}
            query_from_predict = arguments.get("query", "")

            params = {
                "query": query_from_predict,
                "source_filter": "toollist",
                "user_id": 136451106,
                "top_k": 5
            }
            return params
        except Exception as e:
            logger.error(f"从pair1预测中提取检索参数失败: {e}")
            return {}
    
    async def call_retrieval_tool(self, session: aiohttp.ClientSession, params: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """异步调用检索工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": "req_001",
            "method": "tools/call",
            "params": {
                "name": "retrieval_tool",
                "arguments": params,
            },
        }
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(RETRIEVAL_ENDPOINT, headers=RETRIEVAL_HEADERS, json=payload, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    code = resp.status
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"raw": await resp.text()}
                    return code, data
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"检索工具调用失败，第{attempt+1}次尝试: {e}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"检索工具调用失败，已尝试{self.max_retries}次: {e}")
                    return 0, {"error": str(e)}
    
    def extract_retrieved_tools(self, response_obj: Dict[str, Any], top_k: int = 5) -> List[str]:
        """从检索工具响应中提取前top_k个工具名称"""
        tools = []
        
        try:
            # 尝试从result字段中提取工具列表
            if "result" in response_obj and isinstance(response_obj["result"], list):
                for item in response_obj["result"][:top_k]:
                    if isinstance(item, dict):
                        # 尝试不同的字段名
                        for key in ["name", "tool_name", "title", "id", "label", "api_name"]:
                            if key in item and isinstance(item[key], str):
                                tools.append(item[key])
                                break
                        # 如果没找到名称字段，尝试从description或其他字段中提取
                        if not any(key in item for key in ["name", "tool_name", "title", "id", "label", "api_name"]):
                            # 尝试从文本中提取工具名称
                            text = json.dumps(item, ensure_ascii=False)
                            # 简单的启发式方法：查找可能的工具名称模式
                            matches = re.findall(r'"([^"]+)"', text)
                            if matches:
                                tools.append(matches[0])
            
            # 如果result字段没有找到，尝试其他可能的字段
            elif "data" in response_obj and isinstance(response_obj["data"], list):
                for item in response_obj["data"][:top_k]:
                    if isinstance(item, dict):
                        for key in ["name", "tool_name", "title", "id", "label", "api_name"]:
                            if key in item and isinstance(item[key], str):
                                tools.append(item[key])
                                break
            
            # 如果都没有找到，尝试从整个响应中搜索工具名称
            if not tools:
                text = json.dumps(response_obj, ensure_ascii=False)
                # 使用简单的模式匹配来查找可能的工具名称
                matches = re.findall(r'"name":\s*"([^"]+)"', text)
                tools = matches[:top_k]
        
        except Exception as e:
            logger.error(f"提取检索工具时出错: {e}")
        
        return tools[:top_k]  # 确保不超过top_k个
    
    def compute_recall(self, pair1_source: str, pair2_target: str) -> Tuple[int, Dict[str, Any]]:
        """计算recall指标（保留：基于 pair1 source 的原始查询）"""
        try:
            # 提取pair1的查询参数
            params = self.extract_query_params(pair1_source)
            if not params:
                return 0, {"error": "无法提取查询参数"}
            
            # 调用检索工具
            status_code, response = self.call_retrieval_tool(params)
            if status_code != 200:
                return 0, {"error": f"检索工具调用失败，状态码: {status_code}"}
            
            # 提取检索到的工具列表
            retrieved_tools = self.extract_retrieved_tools(response, top_k=5)
            
            # 提取pair2的目标工具名
            try:
                pair2_call = json.loads(pair2_target)
                target_tool = pair2_call.get("name", "")
            except:
                target_tool = ""
            
            # 计算recall
            recall = 1 if target_tool in retrieved_tools else 0
            
            recall_details = {
                "target_tool": target_tool,
                "retrieved_tools": retrieved_tools,
                "recall": recall,
                "query_params": params,
                "response_status": status_code
            }
            
            return recall, recall_details
            
        except Exception as e:
            logger.error(f"计算recall失败: {e}")
            return 0, {"error": str(e)}

    async def compute_recall_from_pair1_predict(self, session: aiohttp.ClientSession, pair1_predict: str, pair2_target: str) -> Tuple[int, Dict[str, Any]]:
        """计算recall指标：基于 pair1 的预测调用中的 query 字段"""
        try:
            params = self.extract_query_params_from_pair1_predict(pair1_predict)
            if not params:
                return 0, {"error": "无法从pair1预测中提取检索参数"}

            logger.info(f"调用检索工具 - 查询参数: {params.get('query', '')[:100]}")
            
            status_code, response = await self.call_retrieval_tool(session, params)
            if status_code != 200:
                logger.warning(f"检索工具调用失败，状态码: {status_code}")
                return 0, {"error": f"检索工具调用失败，状态码: {status_code}"}

            retrieved_tools = self.extract_retrieved_tools(response, top_k=5)
            logger.info(f"检索工具返回 - 获取到 {len(retrieved_tools)} 个工具: {retrieved_tools}")

            try:
                pair2_call = json.loads(pair2_target)
                target_tool = pair2_call.get("name", "")
            except Exception:
                target_tool = ""

            recall = 1 if target_tool in retrieved_tools else 0

            recall_details = {
                "target_tool": target_tool,
                "retrieved_tools": retrieved_tools,
                "recall": recall,
                "query_params": params,
                "response_status": status_code
            }

            return recall, recall_details
        except Exception as e:
            logger.error(f"计算recall失败(基于pair1预测): {e}")
            return 0, {"error": str(e)}

class ToolCallEvaluator:
    """工具调用评估模块：比较tool选择和参数一致性"""
    
    def __init__(self):
        logger.info("初始化工具调用评估模块")
    
    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        """从文本中提取工具调用信息"""
        try:
            # 尝试解析JSON格式的工具调用
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            
            # 尝试从tool_call标签中提取
            tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
            match = re.search(tool_call_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 直接尝试解析整个文本
            return json.loads(text)
        except:
            return {}
    
    def evaluate_tool_call(self, target: str, predict: str) -> Tuple[float, float, Dict[str, Any]]:
        """
        评估工具调用的一致性
        返回：(总分, 工具名称得分, 详细信息)
        """
        target_call = self.extract_tool_call(target)
        predict_call = self.extract_tool_call(predict)
        if not predict_call:
            logger.debug(f"predict 非结构化输出，无法解析为工具调用。predict预览: {predict[:300]}")
        
        details = {
            "target_call": target_call,
            "predict_call": predict_call,
            "tool_name_match": False,
            "arguments_match": False,
            "argument_details": {}
        }
        
        score = 0.0
        tool_name_score = 0.0  # 单独的工具名称得分
        
        # 检查工具名称
        target_name = target_call.get("name", "")
        predict_name = predict_call.get("name", "")
        
        if target_name == predict_name and target_name:
            details["tool_name_match"] = True
            score += 0.5
            tool_name_score = 1.0  # 工具名称完全匹配得满分
        
        # 检查参数
        target_args = target_call.get("arguments", {})
        predict_args = predict_call.get("arguments", {})
        
        if target_args and predict_args:
            matching_args = 0
            total_args = len(target_args)
            
            for key, target_value in target_args.items():
                predict_value = predict_args.get(key)
                match = (predict_value == target_value)
                details["argument_details"][key] = {
                    "target": target_value,
                    "predict": predict_value,
                    "match": match
                }
                if match:
                    matching_args += 1
            
            if total_args > 0:
                arg_score = matching_args / total_args
                details["arguments_match"] = (arg_score == 1.0)
                score += 0.5 * arg_score
        
        return score, tool_name_score, details

class TextGenerationEvaluator:
    """文本生成评估模块：使用LoRA测试模型进行评估"""
    
    def __init__(self, model_type: str = "qwen3"):
        self.model_type = QWEN_MODEL_NAME  # 使用全局配置的模型名称
        self.max_retries = 5
        self.retry_delay = 10
        logger.info(f"初始化文本生成评估模块，使用模型: {self.model_type}")
    
    def call_gemini_api(self, prompt: str, temperature: float = 0.3, top_p: float = 0.95, top_k: int = 40) -> str:
         """调用Gemini API"""
         url = f"https://generativelanguage.googleapis.com/v2beta/models/{self.model_type}:generateContent?key={GEMINI_API_KEY}"
         headers = {"Content-Type": "application/json"}
         payload = {
             "contents": [
                 {
                     "role": "user",
                     "parts": [{"text": prompt}]
                 }
             ],
             "generationConfig": {
                 "temperature": float(temperature),
                 "topP": float(top_p),
                 "topK": int(top_k),
                 "maxOutputTokens": 8192
             }
         }

         for attempt in range(self.max_retries):
             try:
                 response = requests.post(url, headers=headers, json=payload, timeout=60)
                 response.raise_for_status()
                 raw = response.json()
                
                 # 提取文本内容
                 text = ""
                 try:
                     text = raw["candidates"][0]["content"]["parts"][0]["text"]
                 except Exception:
                     text = ""
                
                 return text
                
             except Exception as e:
                 if attempt < self.max_retries - 1:
                     time.sleep(self.retry_delay)
                 else:
                     logger.error(f"API调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                     return ""
    
    async def call_qwen_api(self, session: aiohttp.ClientSession, prompt: List[Dict], temperature: float = 0.3, top_p: float = 0.95) -> str:
        """异步调用Qwen API进行评估"""
        headers = {
            "Content-Type": "application/json"
        }
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
        
        data = {
            "model": self.model_type,
            "messages": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                # 打印完整messages以便完全复现调用
                try:
                    logger.debug(f"LLM 调用完整messages: {json.dumps(data.get('messages', []), ensure_ascii=False) }")
                except Exception:
                    pass
                async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        # 保险起见，去除可能残留的 <think> 块
                    else:
                        error_msg = f"API调用失败，状态码: {response.status}, 响应: {await response.text()}"
                        if attempt < self.max_retries - 1:
                            logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                            await asyncio.sleep(2 ** attempt)  # 指数退避：2^0, 2^1, 2^2秒
                        else:
                            raise Exception(error_msg)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error_msg = f"网络请求异常: {str(e)}"
                if attempt < self.max_retries - 1:
                    logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise Exception(error_msg)
        
        return ""
    
    async def evaluate_text_generation(self, session: aiohttp.ClientSession, target: str, predict: str) -> Tuple[float, Dict[str, Any]]:
        """使用LoRA测试模型评估文本生成质量"""
        judge_prompt = f"""
请评估以下两个文本的相似度和质量，从以下几个维度进行评分（每个维度0-10分）：

1. 内容准确性：预测文本是否准确传达了目标文本的主要信息
2. 完整性：预测文本是否包含了目标文本的关键要素
3. 表达质量：预测文本的语言表达是否清晰、流畅
4. 格式一致性：预测文本的格式是否与目标文本相似

目标文本：
{target}

预测文本：
{predict}

请按以下JSON格式返回评估结果：
{{
    "content_accuracy": <0-10分>,
    "completeness": <0-10分>,
    "expression_quality": <0-10分>,
    "format_consistency": <0-10分>,
    "overall_score": <0-10分>,
    "reasoning": "详细说明评分理由"
}}
"""
        
        try:
            # 构建提示词
            prompt = [
                {"role": "system", "content": "你是一个专业的文本质量评估专家，能够客观地评估文本的相似度和质量。"},
                {"role": "user", "content": judge_prompt}
            ]
            
            # 调用Qwen API进行评估
            result = await self.call_qwen_api(session, prompt, temperature=0.3, top_p=0.95)
            
            # 提取JSON结果
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                eval_result = json.loads(json_match.group())
                overall_score = eval_result.get("overall_score", 0) / 10.0  # 转换为0-1分数
                return overall_score, eval_result
            else:
                # 如果无法解析JSON，尝试简单的文本匹配评分
                logger.warning("无法解析JSON评估结果，使用简单文本匹配评分")
                simple_score = self._simple_text_similarity_score(target, predict)
                return simple_score, {"overall_score": simple_score * 10, "method": "simple_similarity"}
                
        except Exception as e:
            logger.error(f"文本生成评估失败: {e}")
            # 如果评估失败，使用简单的文本相似度评分
            simple_score = self._simple_text_similarity_score(target, predict)
            return simple_score, {"error": str(e), "fallback_score": simple_score * 10}
    
    def _simple_text_similarity_score(self, target: str, predict: str) -> float:
        """简单的文本相似度评分（备用方法）"""
        try:
            # 简单的基于长度和关键词的相似度评分
            target_words = set(target.lower().split())
            predict_words = set(predict.lower().split())
            
            if not target_words:
                return 0.0
            
            # 计算词汇重叠度
            overlap = len(target_words.intersection(predict_words))
            overlap_ratio = overlap / len(target_words)
            
            # 考虑长度相似度
            length_ratio = min(len(predict), len(target)) / max(len(predict), len(target)) if max(len(predict), len(target)) > 0 else 0
            
            # 综合评分（0-1）
            score = (overlap_ratio * 0.7 + length_ratio * 0.3)
            return min(score, 1.0)
            
        except Exception:
            return 0.5  # 默认中等分数

class MetricsCalculator:
    """指标计算模块"""
    
    def __init__(self):
        logger.info("初始化指标计算模块")
    
    def calculate_pair_metrics(self, results: List[EvaluationResult], pair_id: int, metric_type: str) -> Dict[str, float]:
        """计算特定pair和指标类型的统计"""
        # 过滤出指定pair_id的结果
        pair_results = [r for r in results if r.pair_id == pair_id]
        
        if not pair_results:
            return {"total": 0, "tool_call_avg": 0.0, "tool_name_avg": 0.0}
        
        # 根据指标类型过滤
        if metric_type == "current_logic":
            # 当前逻辑指标：所有结果
            filtered_results = pair_results
        elif metric_type == "real_tool":
            # 真实调用工具指标：仅在recall=1条件下计算（仅适用于pair2）
            if pair_id == 2:
                filtered_results = [r for r in pair_results if r.recall == 1]
            else:
                # pair1不涉及recall，返回空结果
                return {"total": 0, "tool_call_avg": 0.0, "tool_name_avg": 0.0}
        elif metric_type == "recall_subset":
            # recall=1子集指标：recall=1的结果（仅适用于pair2）
            if pair_id == 2:
                filtered_results = [r for r in pair_results if r.recall == 1]
            else:
                filtered_results = []  # pair1不涉及recall
        else:
            filtered_results = pair_results
        
        if not filtered_results:
            return {"total": 0, "accuracy": 0.0, "precision@1": 0.0}
        
        total = len(filtered_results)
        accuracy = sum(r.score for r in filtered_results) / total
        precision_at_1 = sum(r.tool_name_score for r in filtered_results) / total
        
        return {
            "total": total,
            "accuracy": accuracy,
            "precision@1": precision_at_1
        }
    
    def calculate_text_generation_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """计算文本生成指标"""
        text_results = [r for r in results if r.pair_type == "text_generation"]
        
        if not text_results:
            return {"total": 0, "answer_score": 0.0}
        
        total = len(text_results)
        answer_score = sum(r.score for r in text_results) / total
        
        return {
            "total": total,
            "answer_score": answer_score
        }
    
    def calculate_recall_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """计算recall指标"""
        # 只考虑pair2的结果
        pair2_results = [r for r in results if r.pair_id == 2 and r.recall is not None]
        
        if not pair2_results:
            return {"total_pairs": 0, "recall@5_1": 0, "recall@5_0": 0, "recall_rate": 0.0}
        
        total_pairs = len(pair2_results)
        recall_at_5_1 = sum(1 for r in pair2_results if r.recall == 1)
        recall_at_5_0 = total_pairs - recall_at_5_1
        recall_rate = recall_at_5_1 / total_pairs if total_pairs > 0 else 0.0
        
        return {
            "total_pairs": total_pairs,
            "recall@5_1": recall_at_5_1,
            "recall@5_0": recall_at_5_0,
            "recall_rate": recall_rate
        }
    
    def calculate_overall_metrics(self, results: List[EvaluationResult], metric_type: str) -> Dict[str, float]:
        """计算总体指标"""
        if metric_type == "current_logic":
            # 当前逻辑指标：所有结果
            filtered_results = results
        elif metric_type == "real_tool":
            # 真实调用工具指标：pair2的recall=1结果 + pair1的所有结果（pair1不涉及recall）
            filtered_results = []
            for r in results:
                if r.pair_id == 2:
                    if r.recall == 1:  # 只有recall=1的pair2结果
                        filtered_results.append(r)
                else:
                    # pair1和pair3的所有结果都包含
                    filtered_results.append(r)
        elif metric_type == "recall_subset":
            # recall=1子集指标：recall=1的pair2结果 + 其他pair
            filtered_results = []
            for r in results:
                if r.pair_id == 2:
                    if r.recall == 1:
                        filtered_results.append(r)
                else:
                    filtered_results.append(r)
        else:
            filtered_results = results
        
        if not filtered_results:
            return {"total": 0, "accuracy": 0.0, "precision@1": 0.0, "answer_score": 0.0}
        
        total = len(filtered_results)
        
        # 分别计算工具调用和文本生成的得分
        tool_call_results = [r for r in filtered_results if r.pair_type == "tool_call"]
        text_gen_results = [r for r in filtered_results if r.pair_type == "text_generation"]
        
        accuracy = sum(r.score for r in tool_call_results) / len(tool_call_results) if tool_call_results else 0.0
        precision_at_1 = sum(r.tool_name_score for r in tool_call_results) / len(tool_call_results) if tool_call_results else 0.0
        answer_score = sum(r.score for r in text_gen_results) / len(text_gen_results) if text_gen_results else 0.0
        
        return {
            "total": total,
            "accuracy": accuracy,
            "precision@1": precision_at_1,
            "answer_score": answer_score
        }
    
    def update_realtime_metrics(self, metrics: RealTimeMetrics, results: List[EvaluationResult]) -> RealTimeMetrics:
        """更新实时指标"""
        # 更新基本统计
        metrics.total_conversations = len(set(r.conversation_id for r in results))
        metrics.total_pairs = len(results)
        
        # 更新pair1指标
        metrics.pair1 = self.calculate_pair_metrics(results, 1, "current_logic")
        
        # 更新pair2指标
        metrics.pair2 = self.calculate_pair_metrics(results, 2, "current_logic")
        metrics.pair2_consider_recall = self.calculate_pair_metrics(results, 2, "real_tool")  # 仅在recall=1条件下计算
        
        # 更新pair3指标（文本生成）
        metrics.pair3 = self.calculate_text_generation_metrics(results)
        
        # 更新recall指标
        metrics.recall_metrics = self.calculate_recall_metrics(results)
        
        # 更新总体指标
        metrics.overall_current_logic = self.calculate_overall_metrics(results, "current_logic")
        
        return metrics

class TrainingDataEvaluator:
    """主评估类"""
    
    def __init__(self, model_type: str = "qwen3"):
        self.data_processor = DataProcessor()
        self.llm_predictor = LLMPredictor(model_type)
        self.tool_evaluator = ToolCallEvaluator()
        self.text_evaluator = TextGenerationEvaluator(model_type)  # 使用LoRA模型而不是Gemini
        self.retrieval_caller = RetrievalToolCaller()
        self.metrics_calculator = MetricsCalculator()
        logger.info("训练数据评估器初始化完成")
    
    async def evaluate_single_pair(self, session: aiohttp.ClientSession, pair: EvaluationPair, pair_predict_by_id: Dict[int, str], pair_toolname_score_by_id: Dict[int, float]) -> EvaluationResult:
        """异步评估单个pair"""
        logger.info(f"评估 Pair {pair.pair_id} (类型: {pair.pair_type})")
        
        try:
            logger.debug(f"Pair {pair.pair_id} source长度: {len(pair.source)}，预览: {pair.source[:400]}")
            logger.debug(f"Pair {pair.pair_id} target长度: {len(pair.target)}，预览: {pair.target[:200]}")
        except Exception:
            pass
        
        # 生成预测
        predict = await self.llm_predictor.predict(session, pair.source, pair.pair_type)
        # 记录该pair的预测，供后续pair使用
        pair_predict_by_id[pair.pair_id] = predict
        
        # 根据类型选择评估方法
        if pair.pair_type == "tool_call":
            score, tool_name_score, details = self.tool_evaluator.evaluate_tool_call(pair.target, predict)
            # 记录该pair的工具名称匹配分
            pair_toolname_score_by_id[pair.pair_id] = tool_name_score
            
            # 确保默认初始化
            recall = None
            recall_details = None
            
            # 只有pair2才计算recall指标；当开关启用时跳过调用检索服务
            if pair.pair_id == 2 and not DISABLE_RECALL:
                pair1_predict = pair_predict_by_id.get(1)
                pair1_toolname_score = pair_toolname_score_by_id.get(1)
                if pair1_predict and pair1_toolname_score == 1.0:
                    recall, recall_details = await self.retrieval_caller.compute_recall_from_pair1_predict(session, pair1_predict, pair.target)
            elif pair.pair_id == 2 and DISABLE_RECALL:
                recall, recall_details = None, None
        else:
            # pair3（文本生成）不涉及recall
            score, details = await self.text_evaluator.evaluate_text_generation(session, pair.target, predict)
            tool_name_score = 0.0
            recall = None
            recall_details = None
        
        result = EvaluationResult(
            conversation_id=pair.conversation_id,
            pair_id=pair.pair_id,
            pair_type=pair.pair_type,
            source=pair.source,
            target=pair.target,
            predict=predict,
            score=score,
            tool_name_score=tool_name_score,
            recall=recall,
            recall_details=recall_details,
            details=details
        )
        
        # 根据类型输出不同的日志信息（与日志字段命名保持一致）
        if pair.pair_type == "tool_call":
            if recall is not None:
                # 提取检索到的工具列表
                retrieved_tools = recall_details.get("retrieved_tools", []) if recall_details else []
                target_tool = recall_details.get("target_tool", "") if recall_details else ""
                logger.info(f"Pair {pair.pair_id} 评估完成，accuracy: {score:.3f}, precision@1: {tool_name_score:.3f}, recall@5: {recall}")
                logger.info(f"Pair {pair.pair_id} 检索详情 - 目标工具: {target_tool}, 检索到的工具: {retrieved_tools}")
            else:
                logger.info(f"Pair {pair.pair_id} 评估完成，accuracy: {score:.3f}, precision@1: {tool_name_score:.3f}")
        else:
            logger.info(f"Pair {pair.pair_id} 评估完成，answer_score: {score:.3f}")
        
        return result
    
    async def evaluate_file(self, file_path: str, checkpoint_file: str = None, start_idx: int = 0, end_idx: Optional[int] = None) -> List[EvaluationResult]:
        """异步并发评估整个文件，支持断点续传和实时指标更新
        
        Args:
            file_path: 要评估的JSON文件路径
            checkpoint_file: 断点文件路径（可选）
            start_idx: 开始评估的对话索引（从0开始）
            end_idx: 结束评估的对话索引（不包含，如果为None则评估到最后）
        """
        logger.info(f"开始异步并发评估文件: {file_path}")
        logger.info(f"并发配置: 最大对话并发数={MAX_CONCURRENT_CONVERSATIONS}, 最大Pair并发数={MAX_CONCURRENT_PAIRS}, 最大API并发数={MAX_CONCURRENT_API_CALLS}")
        
        if start_idx > 0 or end_idx is not None:
            logger.info(f"评估范围: 对话 {start_idx} 到 {end_idx if end_idx else '最后'}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确定实际的结束索引
        total_conversations = len(data)
        if end_idx is None:
            end_idx = total_conversations
        else:
            end_idx = min(end_idx, total_conversations)
        
        # 验证参数
        if start_idx >= total_conversations:
            logger.error(f"起始索引 {start_idx} 超出数据范围 (总共 {total_conversations} 个对话)")
            return []
        
        if start_idx >= end_idx:
            logger.error(f"起始索引 {start_idx} 不能大于等于结束索引 {end_idx}")
            return []
        
        logger.info(f"实际评估范围: 对话 {start_idx} 到 {end_idx-1} (共 {end_idx - start_idx} 个对话)")
        
        # 检查是否有断点文件
        all_results = []
        processed_pairs = set()  # 记录已处理的(conversation_id, pair_id)组合
        conversation_id = 1
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    all_results = [EvaluationResult(**r) for r in checkpoint_data.get("results", [])]
                    processed_pairs = set(tuple(p) for p in checkpoint_data.get("processed_pairs", []))
                    conversation_id = checkpoint_data.get("next_conversation_id", 1)
                    start_idx = len(set(r.conversation_id for r in all_results))  # 从已处理的对话数开始
                    logger.info(f"从断点恢复，已处理 {len(all_results)} 个评估对，conversation_id: {conversation_id}")
            except Exception as e:
                logger.error(f"读取断点文件失败: {e}，将从头开始评估")
                all_results = []
                start_idx = 0
                processed_pairs = set()
                conversation_id = 1
        
        # 初始化实时指标
        realtime_metrics = RealTimeMetrics()
        
        # 创建aiohttp会话和信号量控制并发
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_API_CALLS, limit_per_host=MAX_CONCURRENT_API_CALLS)
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟总超时
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 创建信号量控制并发数
            conversation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERSATIONS)
            pair_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAIRS)
            
            # 创建所有需要处理的对话任务
            conversation_tasks = []
            for idx, conversation_data in enumerate(data[start_idx:end_idx], start=start_idx):
                task = self._evaluate_conversation_async(
                    session, conversation_semaphore, pair_semaphore,
                    conversation_data, idx, conversation_id, processed_pairs
                )
                conversation_tasks.append(task)
                conversation_id += 1
            
            # 并发执行所有对话的评估
            logger.info(f"开始并发评估 {len(conversation_tasks)} 个对话")
            conversation_results = await asyncio.gather(*conversation_tasks, return_exceptions=True)
            
            # 处理结果和异常
            for idx, result in enumerate(conversation_results):
                if isinstance(result, Exception):
                    logger.error(f"对话 {start_idx + idx} 评估失败: {result}")
                else:
                    all_results.extend(result)
                    
                    # 更新实时指标
                    realtime_metrics = self.metrics_calculator.update_realtime_metrics(realtime_metrics, all_results)
                    self._save_realtime_metrics(realtime_metrics)
                    
                    # 保存断点
                    if checkpoint_file:
                        self._save_checkpoint(checkpoint_file, all_results, processed_pairs, start_idx + idx + 1)
        
        logger.info(f"异步并发评估完成，总共处理了 {len(all_results)} 个评估对")
        return all_results
    
    async def _evaluate_conversation_async(self, session: aiohttp.ClientSession, conversation_semaphore: asyncio.Semaphore, 
                                         pair_semaphore: asyncio.Semaphore, conversation_data: Dict, idx: int, 
                                         conversation_id: int, processed_pairs: set) -> List[EvaluationResult]:
        """异步评估单个对话"""
        async with conversation_semaphore:
            logger.info(f"评估对话 {idx + 1} (conversation_id: {conversation_id})")
            
            # 解析pairs
            pairs = self.data_processor.parse_conversations(conversation_data, conversation_id)
            
            # 过滤出未处理的pairs
            unprocessed_pairs = []
            for pair in pairs:
                pair_key = (conversation_id, pair.pair_id)
                if pair_key not in processed_pairs:
                    unprocessed_pairs.append(pair)
                else:
                    logger.info(f"跳过已处理的 Pair {pair.pair_id}")
            
            if not unprocessed_pairs:
                logger.info(f"对话 {conversation_id} 的所有pairs都已处理过")
                return []
            
            # 为每个对话维护独立的预测和得分记录
            pair_predict_by_id = {}
            pair_toolname_score_by_id = {}
            
            # 需要按顺序处理tool_call类型的pairs，因为pair2依赖pair1的结果
            # 先按pair_id排序，确保pair1在pair2之前处理
            sorted_pairs = sorted(unprocessed_pairs, key=lambda p: p.pair_id)
            
            results = []
            text_gen_pairs = []
            
            # 分离tool_call和text_generation类型的pairs
            for pair in sorted_pairs:
                if pair.pair_type == "tool_call":
                    # tool_call类型的pairs需要串行处理，确保依赖关系
                    result = await self._evaluate_single_pair_async(
                        session, pair_semaphore, pair, pair_predict_by_id, pair_toolname_score_by_id
                    )
                    if isinstance(result, Exception):
                        logger.error(f"Pair {pair.pair_id} 评估失败: {result}")
                    else:
                        results.append(result)
                        pair_key = (conversation_id, pair.pair_id)
                        processed_pairs.add(pair_key)
                else:
                    # text_generation类型的pairs收集起来并发处理
                    text_gen_pairs.append(pair)
            
            # 并发处理所有text_generation类型的pairs
            if text_gen_pairs:
                text_gen_tasks = []
                for pair in text_gen_pairs:
                    task = self._evaluate_single_pair_async(
                        session, pair_semaphore, pair, pair_predict_by_id, pair_toolname_score_by_id
                    )
                    text_gen_tasks.append(task)
                
                text_gen_results = await asyncio.gather(*text_gen_tasks, return_exceptions=True)
                
                # 处理text_generation的结果
                for pair, result in zip(text_gen_pairs, text_gen_results):
                    pair_key = (conversation_id, pair.pair_id)
                    if isinstance(result, Exception):
                        logger.error(f"Pair {pair.pair_id} 评估失败: {result}")
                    else:
                        results.append(result)
                        processed_pairs.add(pair_key)
            
            return results
    
    async def _evaluate_single_pair_async(self, session: aiohttp.ClientSession, pair_semaphore: asyncio.Semaphore,
                                        pair: EvaluationPair, pair_predict_by_id: Dict[int, str], 
                                        pair_toolname_score_by_id: Dict[int, float]) -> EvaluationResult:
        """异步评估单个pair（带信号量控制）"""
        async with pair_semaphore:
            return await self.evaluate_single_pair(session, pair, pair_predict_by_id, pair_toolname_score_by_id)
    
    def _save_checkpoint(self, checkpoint_file: str, all_results: List[EvaluationResult], 
                        processed_pairs: set, next_conversation_id: int):
        """保存断点文件"""
        try:
            # 清理pair1和pair3的recall字段
            cleaned_results = []
            for r in all_results:
                result_dict = asdict(r)
                # 对于pair1和pair3，移除recall相关字段
                if r.pair_id in [1, 3]:
                    result_dict.pop('recall', None)
                    result_dict.pop('recall_details', None)
                cleaned_results.append(result_dict)
            
            checkpoint_data = {
                "results": cleaned_results,
                "processed_pairs": [list(p) for p in processed_pairs],
                "next_conversation_id": next_conversation_id
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存断点文件失败: {e}")
    
    def _save_realtime_metrics(self, metrics: RealTimeMetrics):
        """保存实时指标到文件"""
        try:
            realtime_file = "metrics/realtime_metrics.json"
            data = asdict(metrics)
            # 将 overall_current_logic 重命名为 overall
            if "overall_current_logic" in data:
                data["overall"] = data.pop("overall_current_logic")
            # 保留所有数值到 3 位小数
            data = _round_floats(data, 3)
            with open(realtime_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存实时指标失败: {e}")
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """生成评估报告，按pair_id分组"""
        # 按pair_id分组结果
        grouped_results = defaultdict(list)
        for result in results:
            grouped_results[result.pair_id].append(result)
        
        # 计算各种指标
        metrics_calc = MetricsCalculator()
        
        # 按pair分组的指标
        pair_metrics = {}
        for pair_id in [1, 2, 3]:
            pair_results = grouped_results.get(pair_id, [])
            if pair_results:
                if pair_id == 1:
                    # pair1指标
                    pair_metrics["pair1"] = metrics_calc.calculate_pair_metrics(pair_results, pair_id, "current_logic")
                elif pair_id == 2:
                    # pair2指标
                    pair_metrics["pair2"] = metrics_calc.calculate_pair_metrics(pair_results, pair_id, "current_logic")
                    pair_metrics["pair2_consider_recall"] = metrics_calc.calculate_pair_metrics(pair_results, pair_id, "real_tool")
                    pair_metrics["pair2_recall_subset"] = metrics_calc.calculate_pair_metrics(pair_results, pair_id, "recall_subset")
                else:
                    # pair3指标
                    pair_metrics["pair3"] = metrics_calc.calculate_text_generation_metrics(pair_results)
        
        # recall指标
        recall_metrics = metrics_calc.calculate_recall_metrics(results)
        
        # 总体指标
        overall_metrics = metrics_calc.calculate_overall_metrics(results, "current_logic")
        
        # 构建报告
        report = {
            "summary": {
                "total_conversations": len(set(r.conversation_id for r in results)),
                "total_pairs": len(results),
                "pair_metrics": pair_metrics,
                "recall_metrics": recall_metrics,
                "overall_metrics": overall_metrics,
                "model": self.llm_predictor.model_type
            },
            "detailed_results": {
                f"pair{pair_id}": [
                    {
                        "conversation_id": r.conversation_id,
                        "pair_id": r.pair_id,
                        "pair_type": r.pair_type,
                        "score": r.score,
                        "tool_name_score": r.tool_name_score if r.pair_type == "tool_call" else None,
                        **({"recall": r.recall, "recall_details": r.recall_details} if pair_id == 2 and r.recall is not None else {}),
                        "source": r.source,
                        "target": r.target,
                        "predict": r.predict,
                        "target_preview": r.target[:100] + "..." if len(r.target) > 100 else r.target,
                        "predict_preview": r.predict[:100] + "..." if len(r.predict) > 100 else r.predict,
                        "details": r.details
                    }
                    for r in pair_results
                ]
                for pair_id, pair_results in grouped_results.items()
            }
        }
        
        return report

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练数据评估脚本")
    parser.add_argument("--input_file", "-i", type=str, 
                       default="/home/ziqiang/LLaMA-Factory/data/dataset/10_22/10.22_fuzzy_data.json",
                       help="输入JSON文件路径 (默认: data/9.17_evaluate_data_top5_final.json)")
    parser.add_argument("--output_file", "-o", type=str,
                       default="/home/ziqiang/LLaMA-Factory/data/dataset/10_22/data_evaluation.json",
                       help="输出结果文件路径 (默认: metrics/data_evaluation_results.json)")
    parser.add_argument("--checkpoint_file", "-c", type=str,
                       default="/home/ziqiang/LLaMA-Factory/data/dataset/10_22/evaluation_checkpoint.json",
                       help="断点文件路径 (默认: metrics/evaluation_checkpoint.json)")
    parser.add_argument("--start_idx", "-s", type=int, default=0,
                       help="开始评估的对话索引（从0开始，默认: 0）")
    parser.add_argument("--end_idx", "-e", type=int, default=2000,
                       help="结束评估的对话索引（不包含，默认: 10）")
    parser.add_argument("--log_file", "-l", type=str,
                       default="/home/ziqiang/LLaMA-Factory/data/dataset/10_22/data_evaluation.log",
                       help="日志文件路径 (默认: metrics/data_evaluation.log)")
    parser.add_argument("--models", type=str, default="",
                       help="以逗号分隔的一组模型名（例如: /data/models/Qwen3-8B,my_lora）。提供多个时开启多模型评估模式")
    parser.add_argument("--multi_output_dir", type=str, default="evaluation/multi",
                       help="多模型评估输出目录（默认: evaluation/multi）")
    parser.add_argument("--aggregate_output", type=str, default="evaluation/multi_aggregate_0929_v2.json",
                       help="多模型聚合报告输出文件（默认: evaluation/multi_aggregate.json）")
    
    # 并发控制参数
    parser.add_argument("--max_concurrent_conversations", type=int, default=1,
                       help="最大并发对话数（默认: 5）")
    parser.add_argument("--max_concurrent_pairs", type=int, default=1,
                       help="最大并发pair数（默认: 10）")
    parser.add_argument("--max_concurrent_api_calls", type=int, default=1,
                       help="最大并发API调用数（默认: 20）")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 更新全局并发配置
    global MAX_CONCURRENT_CONVERSATIONS, MAX_CONCURRENT_PAIRS, MAX_CONCURRENT_API_CALLS
    MAX_CONCURRENT_CONVERSATIONS = args.max_concurrent_conversations
    MAX_CONCURRENT_PAIRS = args.max_concurrent_pairs
    MAX_CONCURRENT_API_CALLS = args.max_concurrent_api_calls
    
    # 添加日志配置
    logger.add(args.log_file, rotation="100 MB", level="DEBUG")
    
    # 创建输出目录
    os.makedirs("metrics", exist_ok=True)
    
    logger.info("开始增强版异步并发训练数据评估")
    logger.info(f"输入文件: {args.input_file}")
    logger.info(f"输出文件: {args.output_file}")
    logger.info(f"断点文件: {args.checkpoint_file}")
    logger.info(f"评估范围: 对话 {args.start_idx} 到 {args.end_idx if args.end_idx else '最后'}")
    logger.info(f"并发配置: 对话={MAX_CONCURRENT_CONVERSATIONS}, Pairs={MAX_CONCURRENT_PAIRS}, API={MAX_CONCURRENT_API_CALLS}")
    
    # 多模型模式：当 --models 指定多个模型时，循环评估并聚合
    models_list = [m.strip() for m in (args.models or "").split(',') if m.strip()]
    if len(models_list) > 1:
        os.makedirs(args.multi_output_dir, exist_ok=True)
        aggregate = {
            "input_file": args.input_file,
            "models": models_list,
            "runs": {}
        }
        for model_name in models_list:
            model_safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
            output_file = os.path.join(args.multi_output_dir, f"result_{model_safe}.json")
            checkpoint_file = os.path.join(args.multi_output_dir, f"checkpoint_{model_safe}.json")
            log_file = os.path.join(args.multi_output_dir, f"eval_{model_safe}.log")
            try:
                logger.add(log_file, rotation="100 MB", level="DEBUG")
            except Exception:
                pass

            evaluator = TrainingDataEvaluator(model_type=model_name)

            results = await evaluator.evaluate_file(
                args.input_file,
                checkpoint_file,
                args.start_idx,
                args.end_idx
            )
            report = evaluator.generate_report(results)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_round_floats(report, 3), f, ensure_ascii=False, indent=2)
            aggregate["runs"][model_name] = {
                "output_file": output_file,
                "summary": report.get("summary", {}),
            }

            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                except Exception:
                    pass

        # 生成聚合对比（抽取关键指标）
        comparison = {}
        for model_name, run in aggregate["runs"].items():
            summary = run.get("summary", {})
            comparison[model_name] = {
                "overall_metrics": summary.get("overall_metrics", {}),
                "pair1": summary.get("pair_metrics", {}).get("pair1", {}),
                "pair2": summary.get("pair_metrics", {}).get("pair2", {}),
                "pair3": summary.get("pair_metrics", {}).get("pair3", {}),
            }
        aggregate["comparison"] = _round_floats(comparison, 3)

        with open(args.aggregate_output, 'w', encoding='utf-8') as f:
            json.dump(_round_floats(aggregate, 3), f, ensure_ascii=False, indent=2)
        logger.info(f"多模型评估完成，聚合报告: {args.aggregate_output}")
        
        # 确保所有异步任务完成
        import gc
        gc.collect()
        return

    # 单模型模式：保持原有行为
    evaluator = TrainingDataEvaluator(
        model_type=QWEN_MODEL_NAME if not models_list else models_list[0]
    )
    results = await evaluator.evaluate_file(
        args.input_file,
        args.checkpoint_file,
        args.start_idx,
        args.end_idx
    )
    report = evaluator.generate_report(results)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(_round_floats(report, 3), f, ensure_ascii=False, indent=2)
    logger.info(f"评估完成，结果已保存到: {args.output_file}")
    if os.path.exists(args.checkpoint_file):
        try:
            os.remove(args.checkpoint_file)
            logger.info(f"已删除断点文件: {args.checkpoint_file}")
        except Exception as e:
            logger.error(f"删除断点文件失败: {e}")
    
    # 确保所有异步任务完成
    import gc
    gc.collect()
    
    

if __name__ == "__main__":
    asyncio.run(main())