#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多跳模型评估脚本（与训练template完全匹配）
使用qwen3 template格式：<|im_start|>...<|im_end|>
适配cutoff_len=10240和完整tools定义方案
"""

import json
import asyncio
import re
import sys
import os
import time
import argparse
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from pathlib import Path
from collections import defaultdict
import aiohttp

# 递归保留浮点到小数点后三位
def _round_floats(obj: Any, ndigits: int = 3) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    return obj

# 模型服务配置
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:5526")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "my_lora")   # 需要确认实际模型名
QWEN_API_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"

# Retrieval Tool API 配置（可选，用于计算recall指标）
RETRIEVAL_ENDPOINT = os.getenv("RETRIEVAL_ENDPOINT", "http://125.122.38.32:8084/v1/databoard/tools/call")
RETRIEVAL_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

# 临时开关：跳过调用检索服务与相关recall指标统计
DISABLE_RECALL = str(os.getenv("EVAL_DISABLE_RECALL", "0")).lower() in ("1", "true", "yes")  # 默认禁用

# 并发控制配置
MAX_CONCURRENT_CONVERSATIONS = int(os.getenv("MAX_CONCURRENT_CONVERSATIONS", "2"))
MAX_CONCURRENT_PAIRS = int(os.getenv("MAX_CONCURRENT_PAIRS", "5"))
MAX_CONCURRENT_API_CALLS = int(os.getenv("MAX_CONCURRENT_API_CALLS", "10"))

@dataclass
class MultiHopPair:
    """多跳评估对结构"""
    pair_id: int
    hop_index: int  # 第几跳（从1开始）
    hop_type: str   # 'retrieval' 或 'business_tool'
    source: str     # 输入（包含累积的上下文）
    target: str     # 期望的输出
    conversation_id: int
    
    # 上下文历史（用于累积）
    original_query: str = ""
    observation_history: List[str] = None  # 前面所有的observation

@dataclass
class MultiHopResult:
    """多跳评估结果"""
    conversation_id: int
    pair_id: int
    hop_index: int
    hop_type: str
    source: str
    target: str
    predict: str
    score: float
    tool_name_score: float
    recall: Optional[int] = None  # recall指标（0或1），仅用于retrieval跳
    recall_details: Optional[Dict[str, Any]] = None  # recall详细信息
    details: Dict[str, Any] = None

@dataclass
class MultiHopMetrics:
    """多跳指标结构"""
    total_conversations: int = 0
    total_pairs: int = 0
    
    # 按跳数分组的指标
    by_hop: Dict[int, Dict[str, float]] = None  # {hop_index: {accuracy, precision@1, count}}
    
    # 按类型分组的指标
    retrieval_hops: Dict[str, float] = None  # 所有检索跳的汇总指标
    business_hops: Dict[str, float] = None   # 所有业务工具跳的汇总指标
    
    # 总体指标
    overall: Dict[str, float] = None
    
    def __post_init__(self):
        if self.by_hop is None:
            self.by_hop = {}
        if self.retrieval_hops is None:
            self.retrieval_hops = {"total": 0, "accuracy": 0.0, "precision@1": 0.0}
        if self.business_hops is None:
            self.business_hops = {"total": 0, "accuracy": 0.0, "precision@1": 0.0}
        if self.overall is None:
            self.overall = {"total": 0, "accuracy": 0.0, "precision@1": 0.0}

class MultiHopDataProcessor:
    """多跳数据处理模块：识别多跳模式并构建带上下文累积的pairs"""
    
    def __init__(self):
        logger.info("初始化多跳数据处理模块")
    
    def parse_conversations(self, conversation_data: Dict, conversation_id: int) -> List[MultiHopPair]:
        """
        解析多跳conversations，使用与训练时完全一致的qwen3 template格式
        
        多跳模式识别：
        - human -> function_call (retrieval/business) -> observation -> function_call -> ...
        
        格式规则（与training template完全匹配）：
        - Pair 1: <|im_start|>system\n{system_with_tools}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n
        - Pair N: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<tool_response>\n{observation_content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
        """
        conversations = conversation_data["conversations"]
        system_prompt = conversation_data["system"]
        tools = conversation_data.get("tools", "[]")
        
        pairs = []
        pair_id = 1
        hop_index = 1
        
        # 提取原始用户query
        original_query = ""
        for msg in conversations:
            if msg["from"] == "human":
                original_query = msg["value"]
                break
        
        # 准备system内容（包含tools，用于Pair 1）
        try:
            tools_str = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
        except Exception:
            tools_str = str(tools)

        # 填充tools到system中（Pair 1使用）
        if '<tools>' in system_prompt and '</tools>' in system_prompt:
            base_system_with_tools = re.sub(
                r'<tools>\s*[\s\S]*?</tools>', 
                f'<tools>\n{tools_str}\n</tools>', 
                system_prompt
            )
        else:
            base_system_with_tools = system_prompt
        
        # 累积的observation历史（保留用于可能的扩展）
        observation_history = []
        
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            
            if msg["from"] == "human":
                # Pair 1: human -> function_call，使用qwen3 template格式
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "function_call":
                    # 使用qwen3格式：包含完整tools的system + user问题
                    source = f"<|im_start|>system\n{base_system_with_tools}<|im_end|>\n<|im_start|>user\n{msg['value']}<|im_end|>\n<|im_start|>assistant\n"
                    target = conversations[i + 1]["value"]
                    
                    # 判断是检索跳还是业务跳
                    try:
                        target_obj = json.loads(target)
                        hop_type = "retrieval" if target_obj.get("name") == "retrieval_tool" else "business_tool"
                    except:
                        hop_type = "unknown"
                    
                    pairs.append(MultiHopPair(
                        pair_id=pair_id,
                        hop_index=hop_index,
                        hop_type=hop_type,
                        source=source,
                        target=target,
                        conversation_id=conversation_id,
                        original_query=original_query,
                        observation_history=observation_history.copy()
                    ))
                    pair_id += 1
                    hop_index += 1
                    i += 2
                else:
                    i += 1
            
            elif msg["from"] == "observation":
                # 记录observation到历史（保留用于可能的扩展）
                observation_history.append(msg["value"])
                
                # 后续pairs: observation -> function_call
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "function_call":
                    # 检查下一跳是否需要retrieval_tool
                    next_is_retrieval = False
                    try:
                        next_call = json.loads(conversations[i + 1]["value"])
                        next_is_retrieval = (next_call.get("name") == "retrieval_tool")
                    except:
                        pass
                    
                    # 构建observation内容（与template.py的enhanced_content逻辑完全一致）
                    observation_content = f"用户查询: {original_query}\n\n当前工具返回结果: {msg['value']}"
                    
                    # 如果下一跳是retrieval，添加完整工具定义（与template.py一致）
                    if next_is_retrieval and tools:
                        observation_content = f"[可用工具定义]\n{tools_str}\n\n{observation_content}"
                    
                    # 使用qwen3的format_observation格式：user角色 + <tool_response>标签
                    source = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<tool_response>\n{observation_content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
                    target = conversations[i + 1]["value"]
                    
                    # 判断跳类型
                    try:
                        target_obj = json.loads(target)
                        hop_type = "retrieval" if target_obj.get("name") == "retrieval_tool" else "business_tool"
                    except:
                        hop_type = "unknown"
                    
                    pairs.append(MultiHopPair(
                        pair_id=pair_id,
                        hop_index=hop_index,
                        hop_type=hop_type,
                        source=source,
                        target=target,
                        conversation_id=conversation_id,
                        original_query=original_query,
                        observation_history=observation_history.copy()
                    ))
                    pair_id += 1
                    hop_index += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        logger.info(f"解析出 {len(pairs)} 个多跳评估对 (conversation_id: {conversation_id})")
        logger.info(f"跳数分布: {[(p.hop_index, p.hop_type) for p in pairs]}")
        return pairs

class LLMPredictor:
    """LLM预测模块"""
    
    def __init__(self, model_type: str = "qwen3"):
        self.model_type = QWEN_MODEL_NAME
        self.max_retries = 5
        self.retry_delay = 10
        logger.info(f"初始化LLM预测模块，使用模型: {self.model_type}")
    
    async def call_qwen_api(self, session: aiohttp.ClientSession, prompt: List[Dict], temperature: float = 0.0, top_p: float = 1.0) -> str:
        """异步调用Qwen API"""
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
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        try:
                            content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
                        except Exception:
                            pass
                        logger.debug(f"LLM 返回: {content[:400]}")
                        return content.strip()
                    else:
                        error_msg = f"API调用失败，状态码: {response.status}, 响应: {await response.text()}"
                        if attempt < self.max_retries - 1:
                            logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                            await asyncio.sleep(2 ** attempt)
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
    
    async def predict(self, session: aiohttp.ClientSession, source: str, hop_type: str) -> str:
        """根据source生成预测 - 从qwen3格式中提取system和user内容"""
        try:
            # 提取system部分（从qwen3 template格式中）
            system_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', source, re.DOTALL)
            system_content = system_match.group(1) if system_match else ""
            
            # 提取user部分（qwen3的observation也用user角色）
            user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', source, re.DOTALL)
            user_content = user_match.group(1) if user_match else ""

            prompt = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            logger.info(f"LLM prompt构建完成，hop_type: {hop_type}")
            result = await self.call_qwen_api(session, prompt, temperature=0.0, top_p=1.0)
            logger.info(f"LLM 输出长度: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"LLM预测失败: {e}")
            return ""

class RetrievalToolCaller:
    """检索工具调用模块（可选，用于计算recall指标）"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 2
        logger.info("初始化检索工具调用模块")
    
    def _extract_tool_call_from_text(self, text: str) -> Dict[str, Any]:
        """从模型预测文本中提取工具调用对象"""
        try:
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            match = re.search(r'<tool_call>\s*({[\s\S]*?})\s*</tool_call>', text)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except Exception:
            return {}
    
    def extract_query_from_predict(self, predict: str) -> str:
        """从预测中提取query参数"""
        try:
            call_obj = self._extract_tool_call_from_text(predict)
            arguments = call_obj.get("arguments", {}) if isinstance(call_obj, dict) else {}
            return arguments.get("query", "")
        except Exception as e:
            logger.error(f"提取query失败: {e}")
            return ""
    
    async def call_retrieval_tool(self, session: aiohttp.ClientSession, query: str, user_id: int = 13) -> Tuple[int, Dict[str, Any]]:
        """异步调用检索工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": "req_multihop_eval",
            "method": "tools/call",
            "params": {
                "name": "retrieval_tool",
                "arguments": {
                    "query": query,
                    "source_filter": "toollist",
                    "user_id": str(user_id),
                    "top_k": 5,
                    "trace_id": "trace_multihop_eval"
                },
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
            if "result" in response_obj:
                result = response_obj["result"]
                
                # 情况1: result 是列表
                if isinstance(result, list):
                    for item in result[:top_k]:
                        if isinstance(item, dict):
                            for key in ["name", "tool_name", "title", "id", "label", "api_name"]:
                                if key in item and isinstance(item[key], str):
                                    tools.append(item[key])
                                    break
                
                # 情况2: result 是字典且包含tools字段
                elif isinstance(result, dict) and "tools" in result:
                    tools_list = result.get("tools", [])
                    for item in tools_list[:top_k]:
                        if isinstance(item, dict):
                            for key in ["name", "tool_name", "title", "id", "label", "api_name"]:
                                if key in item and isinstance(item[key], str):
                                    tools.append(item[key])
                                    break
        except Exception as e:
            logger.error(f"提取检索工具时出错: {e}")
        
        return tools[:top_k]
    
    async def compute_recall(self, session: aiohttp.ClientSession, retrieval_predict: str, next_hop_target: str) -> Tuple[int, Dict[str, Any]]:
        """
        计算recall指标：检查下一跳的目标工具是否在检索返回的top5中
        
        Args:
            retrieval_predict: retrieval_tool的预测调用
            next_hop_target: 下一跳的目标工具调用
        """
        try:
            # 从retrieval预测中提取query
            query = self.extract_query_from_predict(retrieval_predict)
            if not query:
                return 0, {"error": "无法从retrieval预测中提取query"}
            
            logger.info(f"调用检索工具 - 查询: {query[:100]}")
            
            # 调用检索服务
            status_code, response = await self.call_retrieval_tool(session, query)
            if status_code != 200:
                logger.warning(f"检索工具调用失败，状态码: {status_code}")
                return 0, {"error": f"检索工具调用失败，状态码: {status_code}"}
            
            # 提取检索到的工具列表
            retrieved_tools = self.extract_retrieved_tools(response, top_k=5)
            logger.info(f"检索工具返回 - 获取到 {len(retrieved_tools)} 个工具: {retrieved_tools}")
            
            # 提取下一跳的目标工具名
            try:
                target_call = json.loads(next_hop_target)
                target_tool = target_call.get("name", "")
            except Exception:
                target_tool = ""
            
            # 计算recall
            recall = 1 if target_tool in retrieved_tools else 0
            
            recall_details = {
                "query": query,
                "target_tool": target_tool,
                "retrieved_tools": retrieved_tools,
                "recall": recall,
                "response_status": status_code
            }
            
            logger.info(f"Recall计算完成 - 目标工具: {target_tool}, Recall@5: {recall}")
            
            return recall, recall_details
        except Exception as e:
            logger.error(f"计算recall失败: {e}")
            return 0, {"error": str(e)}

class ToolCallEvaluator:
    """工具调用评估模块"""
    
    def __init__(self):
        logger.info("初始化工具调用评估模块")
    
    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        """从文本中提取工具调用"""
        try:
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            
            tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
            match = re.search(tool_call_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            return json.loads(text)
        except:
            return self._extract_kv_pairs(text)

    def _extract_kv_pairs(self, text: str) -> Dict[str, Any]:
        """提取简化键值对"""
        try:
            inner = text
            mc = re.search(r'<tool_call>\s*([\s\S]*?)\s*</tool_call>', text)
            if mc:
                inner = mc.group(1)

            pairs = re.findall(r'"([^"]+)"\s*:\s*"([\s\S]*?)"(?=,|$)', inner)
            if not pairs:
                pairs = re.findall(r"'([^']+)'\s*:\s*'([\s\S]*?)'(?=,|$)", inner)

            result: Dict[str, Any] = {}
            for k, v in pairs:
                result[k.strip()] = v.strip()
            return result
        except Exception:
            return {}
    
    def evaluate_tool_call(self, target: str, predict: str) -> Tuple[float, float, Dict[str, Any]]:
        """评估工具调用"""
        # 忽略的元数据字段（不影响评估分数）
        IGNORED_FIELDS = {"user_id", "trace_id", "top_k"}
        
        target_call = self.extract_tool_call(target)
        predict_call = self.extract_tool_call(predict)
        
        # 添加调试日志
        if not predict_call:
            logger.warning(f"❌ 无法解析预测为工具调用，predict预览: {predict[:200]}")
        if not target_call:
            logger.warning(f"❌ 无法解析目标为工具调用，target预览: {target[:200]}")
        
        # 如果解析成功，显示工具名对比
        if target_call and predict_call:
            target_name = target_call.get("name", "未知")
            predict_name = predict_call.get("name", "未知")
            if target_name != predict_name:
                logger.warning(f"⚠️  工具名不匹配 - 目标: {target_name}, 预测: {predict_name}")
            else:
                logger.info(f"✓ 工具名匹配: {target_name}")
        
        details = {
            "target_call": target_call,
            "predict_call": predict_call,
            "tool_name_match": False,
            "arguments_match": False,
            "argument_details": {}
        }
        
        score = 0.0
        tool_name_score = 0.0
        
        if "name" in target_call or "arguments" in target_call:
            # 标准格式
            target_name = target_call.get("name", "")
            predict_name = predict_call.get("name", "")
            if target_name == predict_name and target_name:
                details["tool_name_match"] = True
                score += 0.5
                tool_name_score = 1.0

            target_args = target_call.get("arguments", {}) or {}
            predict_args = predict_call.get("arguments", {}) or {}
            
            # 过滤掉忽略的字段
            target_args_filtered = {k: v for k, v in target_args.items() if k not in IGNORED_FIELDS}
            
            if target_args_filtered and predict_args:
                matching_args = 0
                total_args = len(target_args_filtered)
                for key, target_value in target_args_filtered.items():
                    predict_value = predict_args.get(key)
                    match = (predict_value == target_value)
                    details["argument_details"][key] = {
                        "target": target_value,
                        "predict": predict_value,
                        "match": match
                    }
                    if match:
                        matching_args += 1
                
                # 记录被忽略的字段（仅用于调试，不影响分数）
                for key in IGNORED_FIELDS:
                    if key in target_args:
                        details["argument_details"][key] = {
                            "target": target_args.get(key),
                            "predict": predict_args.get(key),
                            "match": "ignored",
                            "note": "此字段不影响评估分数"
                        }
                
                if total_args > 0:
                    arg_score = matching_args / total_args
                    details["arguments_match"] = (arg_score == 1.0)
                    score += 0.5 * arg_score
                elif len(target_args) == 0 or all(k in IGNORED_FIELDS for k in target_args.keys()):
                    # 如果所有字段都被忽略，则参数部分得满分
                    details["arguments_match"] = True
                    score += 0.5
        else:
            # 简化格式
            keys_to_check = list(target_call.keys())
            if keys_to_check:
                matching = 0
                for key in keys_to_check:
                    tv = target_call.get(key)
                    pv = predict_call.get(key)
                    is_match = (pv == tv and pv is not None)
                    details["argument_details"][key] = {
                        "target": tv,
                        "predict": pv,
                        "match": is_match
                    }
                    if is_match:
                        matching += 1
                arg_score = matching / len(keys_to_check)
                details["arguments_match"] = (arg_score == 1.0)
                score = arg_score
                sf_target = target_call.get("source_filter")
                sf_predict = predict_call.get("source_filter")
                tool_name_score = 1.0 if (sf_target and sf_target == sf_predict) else 0.0
        
        return score, tool_name_score, details

class MetricsCalculator:
    """多跳指标计算模块"""
    
    def __init__(self):
        logger.info("初始化多跳指标计算模块")
    
    def calculate_metrics(self, results: List[MultiHopResult]) -> MultiHopMetrics:
        """计算多跳指标"""
        metrics = MultiHopMetrics()
        
        if not results:
            return metrics
        
        metrics.total_conversations = len(set(r.conversation_id for r in results))
        metrics.total_pairs = len(results)
        
        # 按跳数分组统计
        by_hop = defaultdict(lambda: {"total": 0, "accuracy": 0.0, "precision@1": 0.0, "scores": [], "tool_name_scores": []})
        for r in results:
            by_hop[r.hop_index]["total"] += 1
            by_hop[r.hop_index]["scores"].append(r.score)
            by_hop[r.hop_index]["tool_name_scores"].append(r.tool_name_score)
        
        for hop_idx, data in by_hop.items():
            data["accuracy"] = sum(data["scores"]) / data["total"] if data["total"] > 0 else 0.0
            data["precision@1"] = sum(data["tool_name_scores"]) / data["total"] if data["total"] > 0 else 0.0
            del data["scores"]
            del data["tool_name_scores"]
        
        metrics.by_hop = dict(by_hop)
        
        # 按类型分组统计
        retrieval_results = [r for r in results if r.hop_type == "retrieval"]
        business_results = [r for r in results if r.hop_type == "business_tool"]
        
        if retrieval_results:
            metrics.retrieval_hops = {
                "total": len(retrieval_results),
                "accuracy": sum(r.score for r in retrieval_results) / len(retrieval_results),
                "precision@1": sum(r.tool_name_score for r in retrieval_results) / len(retrieval_results)
            }
        
        if business_results:
            metrics.business_hops = {
                "total": len(business_results),
                "accuracy": sum(r.score for r in business_results) / len(business_results),
                "precision@1": sum(r.tool_name_score for r in business_results) / len(business_results)
            }
        
        # 总体指标
        metrics.overall = {
            "total": len(results),
            "accuracy": sum(r.score for r in results) / len(results),
            "precision@1": sum(r.tool_name_score for r in results) / len(results)
        }
        
        return metrics

class MultiHopEvaluator:
    """多跳评估主类"""
    
    def __init__(self, model_type: str = "qwen3"):
        self.data_processor = MultiHopDataProcessor()
        self.llm_predictor = LLMPredictor(model_type)
        self.tool_evaluator = ToolCallEvaluator()
        self.retrieval_caller = RetrievalToolCaller() if not DISABLE_RECALL else None
        self.metrics_calculator = MetricsCalculator()
        logger.info(f"多跳评估器初始化完成（Recall计算: {'启用' if not DISABLE_RECALL else '禁用'}）")
    
    async def evaluate_single_pair(self, session: aiohttp.ClientSession, pair: MultiHopPair) -> MultiHopResult:
        """评估单个pair"""
        logger.info(f"评估 Pair {pair.pair_id} (Hop {pair.hop_index}, 类型: {pair.hop_type})")
        
        # 生成预测
        predict = await self.llm_predictor.predict(session, pair.source, pair.hop_type)
        
        # 评估
        score, tool_name_score, details = self.tool_evaluator.evaluate_tool_call(pair.target, predict)
        
        result = MultiHopResult(
            conversation_id=pair.conversation_id,
            pair_id=pair.pair_id,
            hop_index=pair.hop_index,
            hop_type=pair.hop_type,
            source=pair.source,
            target=pair.target,
            predict=predict,
            score=score,
            tool_name_score=tool_name_score,
            details=details
        )
        
        logger.info(f"Pair {pair.pair_id} 完成，accuracy: {score:.3f}, precision@1: {tool_name_score:.3f}")
        return result
    
    async def evaluate_file(self, file_path: str, checkpoint_file: str = None, start_idx: int = 0, end_idx: Optional[int] = None) -> List[MultiHopResult]:
        """评估整个文件"""
        logger.info(f"开始评估文件: {file_path}")
        logger.info(f"并发配置: 对话={MAX_CONCURRENT_CONVERSATIONS}, Pairs={MAX_CONCURRENT_PAIRS}, API={MAX_CONCURRENT_API_CALLS}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_conversations = len(data)
        if end_idx is None:
            end_idx = total_conversations
        else:
            end_idx = min(end_idx, total_conversations)
        
        if start_idx >= total_conversations:
            logger.error(f"起始索引 {start_idx} 超出范围")
            return []
        
        logger.info(f"评估范围: 对话 {start_idx} 到 {end_idx-1} (共 {end_idx - start_idx} 个)")
        
        all_results = []
        
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_API_CALLS, limit_per_host=MAX_CONCURRENT_API_CALLS)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            conversation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CONVERSATIONS)
            pair_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAIRS)
            
            conversation_tasks = []
            for idx, conversation_data in enumerate(data[start_idx:end_idx], start=start_idx):
                task = self._evaluate_conversation_async(
                    session, conversation_semaphore, pair_semaphore,
                    conversation_data, idx, idx + 1
                )
                conversation_tasks.append(task)
            
            logger.info(f"开始并发评估 {len(conversation_tasks)} 个对话")
            conversation_results = await asyncio.gather(*conversation_tasks, return_exceptions=True)
            
            for idx, result in enumerate(conversation_results):
                if isinstance(result, Exception):
                    logger.error(f"对话 {start_idx + idx} 评估失败: {result}")
                else:
                    all_results.extend(result)
        
        logger.info(f"评估完成，总共处理了 {len(all_results)} 个评估对")
        return all_results
    
    async def _evaluate_conversation_async(self, session: aiohttp.ClientSession, conversation_semaphore: asyncio.Semaphore, 
                                         pair_semaphore: asyncio.Semaphore, conversation_data: Dict, idx: int, 
                                         conversation_id: int) -> List[MultiHopResult]:
        """异步评估单个对话"""
        async with conversation_semaphore:
            logger.info(f"评估对话 {idx + 1} (conversation_id: {conversation_id})")
            
            pairs = self.data_processor.parse_conversations(conversation_data, conversation_id)
            
            if not pairs:
                logger.warning(f"对话 {conversation_id} 没有生成评估对")
                return []
            
            # 多跳pairs必须按顺序评估（因为有上下文依赖和recall计算）
            results = []
            prev_result = None  # 记录上一跳的结果
            
            for pair in pairs:
                async with pair_semaphore:
                    try:
                        result = await self.evaluate_single_pair(session, pair)
                        
                        # 计算recall：如果上一跳是retrieval且当前跳是business_tool
                        if (not DISABLE_RECALL and self.retrieval_caller and 
                            prev_result and prev_result.hop_type == "retrieval" and 
                            pair.hop_type == "business_tool" and 
                            prev_result.tool_name_score == 1.0):  # 只有上一跳完全正确才计算recall
                            
                            try:
                                recall, recall_details = await self.retrieval_caller.compute_recall(
                                    session, 
                                    prev_result.predict,  # 上一跳的retrieval预测
                                    pair.target  # 当前跳的目标
                                )
                                result.recall = recall
                                result.recall_details = recall_details
                            except Exception as e:
                                logger.error(f"计算recall失败: {e}")
                        
                        results.append(result)
                        prev_result = result
                    except Exception as e:
                        logger.error(f"Pair {pair.pair_id} 评估失败: {e}")
                        prev_result = None
            
            return results
    
    def generate_report(self, results: List[MultiHopResult]) -> Dict[str, Any]:
        """生成评估报告"""
        # 按对话分组
        conv_groups: Dict[int, List[MultiHopResult]] = defaultdict(list)
        for r in results:
            conv_groups[r.conversation_id].append(r)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_metrics(results)
        
        # 构建cases
        cases = []
        for conv_id, conv_results in sorted(conv_groups.items(), key=lambda x: x[0]):
            hops_data = []
            for r in sorted(conv_results, key=lambda x: x.hop_index):
                hop_entry = {
                    "hop_index": r.hop_index,
                    "hop_type": r.hop_type,
                    "pair_id": r.pair_id,
                    "score": r.score,
                    "tool_name_score": r.tool_name_score,
                    "target_preview": r.target[:100] + "..." if len(r.target) > 100 else r.target,
                    "predict_preview": r.predict[:100] + "..." if len(r.predict) > 100 else r.predict,
                    "details": r.details
                }
                # 添加recall信息（如果有）
                if r.recall is not None:
                    hop_entry["recall"] = r.recall
                    hop_entry["recall_details"] = r.recall_details
                hops_data.append(hop_entry)
            
            case_entry = {
                "conversation_id": conv_id,
                "total_hops": len(conv_results),
                "hops": hops_data
            }
            cases.append(case_entry)
        
        # 计算recall指标
        recall_stats = None
        if not DISABLE_RECALL:
            recall_results = [r for r in results if r.recall is not None]
            if recall_results:
                recall_stats = {
                    "total_with_recall": len(recall_results),
                    "recall@5_count": sum(1 for r in recall_results if r.recall == 1),
                    "recall@5_rate": sum(r.recall for r in recall_results) / len(recall_results) if recall_results else 0.0
                }
        
        report = {
            "summary": {
                "total_conversations": metrics.total_conversations,
                "total_pairs": metrics.total_pairs,
                "by_hop_metrics": metrics.by_hop,
                "retrieval_hops": metrics.retrieval_hops,
                "business_hops": metrics.business_hops,
                "overall_metrics": metrics.overall,
                "recall_metrics": recall_stats,  # 添加recall指标
                "model": self.llm_predictor.model_type,
                "cutoff_len": 10240,
                "template": "qwen3_with_full_tools",
                "recall_enabled": not DISABLE_RECALL
            },
            "cases": cases
        }
        
        return report

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多跳模型评估脚本")
    parser.add_argument("--input_file", "-i", type=str, 
                       default="data/multihop_retrieval_test.json",
                       help="输入JSON文件路径")
    parser.add_argument("--output_file", "-o", type=str,
                       default="evaluation/multihop_evaluation_results.json",
                       help="输出结果文件路径")
    parser.add_argument("--start_idx", "-s", type=int, default=0,
                       help="开始评估的对话索引")
    parser.add_argument("--end_idx", "-e", type=int, default=1000,  
                       help="结束评估的对话索引")
    parser.add_argument("--log_file", "-l", type=str,
                       default="evaluation/multihop_evaluation.log",
                       help="日志文件路径")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 更新全局并发配置
    global MAX_CONCURRENT_CONVERSATIONS, MAX_CONCURRENT_PAIRS, MAX_CONCURRENT_API_CALLS
    
    # 创建输出目录
    os.makedirs("evaluation", exist_ok=True)
    
    # 配置日志
    logger.add(args.log_file, rotation="100 MB", level="DEBUG")
    
    logger.info("=" * 60)
    logger.info("开始多跳模型评估")
    logger.info("=" * 60)
    logger.info(f"模型服务: {VLLM_BASE_URL}")
    logger.info(f"模型名称: {QWEN_MODEL_NAME}")
    logger.info(f"输入文件: {args.input_file}")
    logger.info(f"输出文件: {args.output_file}")
    logger.info(f"评估范围: 对话 {args.start_idx} 到 {args.end_idx}")
    logger.info(f"Template格式: qwen3 template + 完整tools定义")
    logger.info(f"Cutoff长度: 10240")
    logger.info(f"并发配置: 对话={MAX_CONCURRENT_CONVERSATIONS}, Pairs={MAX_CONCURRENT_PAIRS}, API={MAX_CONCURRENT_API_CALLS}")
    
    evaluator = MultiHopEvaluator(model_type=QWEN_MODEL_NAME)
    
    results = await evaluator.evaluate_file(
        args.input_file,
        None,
        args.start_idx,
        args.end_idx
    )
    
    report = evaluator.generate_report(results)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(_round_floats(report, 3), f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"评估完成！结果已保存到: {args.output_file}")
    logger.info("=" * 60)
    logger.info(f"总对话数: {report['summary']['total_conversations']}")
    logger.info(f"总评估对数: {report['summary']['total_pairs']}")
    logger.info(f"总体准确率: {report['summary']['overall_metrics']['accuracy']:.3f}")
    logger.info(f"总体precision@1: {report['summary']['overall_metrics']['precision@1']:.3f}")
    
    # 显示recall指标（如果启用）
    if report['summary'].get('recall_enabled') and report['summary'].get('recall_metrics'):
        recall_metrics = report['summary']['recall_metrics']
        logger.info("=" * 60)
        logger.info("📊 Recall@5 指标:")
        logger.info(f"  - 计算样本数: {recall_metrics['total_with_recall']}")
        logger.info(f"  - 召回成功数: {recall_metrics['recall@5_count']}")
        logger.info(f"  - Recall@5: {recall_metrics['recall@5_rate']:.3f} ({recall_metrics['recall@5_rate']*100:.1f}%)")
    elif not report['summary'].get('recall_enabled'):
        logger.info("=" * 60)
        logger.info("ℹ️  Recall计算已禁用 (DISABLE_RECALL=True)")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

