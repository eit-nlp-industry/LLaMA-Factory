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
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from pathlib import Path
from collections import defaultdict

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

# Qwen API 配置
QWEN_API_URL = "http://125.122.38.32:8021/v1/chat/completions"
QWEN_MODEL_NAME = "/data/models/Qwen3-8B"

# Retrieval Tool API 配置
RETRIEVAL_ENDPOINT = "http://125.122.38.32:8084/v1/databoard/tools/call"
RETRIEVAL_HEADERS = {
    "Token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJuaWNrbmFtZSI6IkpvaG4iLCJyb2xlX2lkIjoxfQ.NFkENyI182Q3XPcGiTOCXWN21ZQoDP40SRGHYQ25vVw",
    "Content-Type": "application/json",
}

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
        
        # 使用原始的system prompt，并插入tools
        if '<tools>' in system_prompt and '</tools>' in system_prompt:
            # 替换空的tools标签为实际的tools内容
            base_system = system_prompt.replace('<tools>\n</tools>', f'<tools>\n{tools}\n</tools>')
        else:
            # 如果没有tools标签，直接使用原始system
            base_system = system_prompt
        
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            
            if msg["from"] == "human":
                # Pair 1: system + tools + user -> function_call
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "function_call":
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
                        # Pair 2: system + tools + user + observation -> function_call
                        source = f"{base_system}\n\nUser: {original_query}\n\nTool Response: {msg['value']}"
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
                        # Pair 3: system + tools + user + observation -> gpt
                        source = f"{base_system}\n\nUser: {original_query}\n\nTool Response: {msg['value']}"
                        target = next_msg["value"]
                        pairs.append(EvaluationPair(
                            pair_id=pair_id,
                            source=source,
                            target=target,
                            pair_type="text_generation",
                            conversation_id=conversation_id
                        ))
                        pair_id += 1
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
        self.max_retries = 3
        self.retry_delay = 5
        logger.info(f"初始化LLM预测模块，使用模型: {self.model_type}")
    
    def call_qwen_api(self, prompt: List[Dict], temperature: float = 0.0, top_p: float = 1.0) -> str:
        """调用Qwen API生成预测"""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_type,
            "messages": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            # 对于多数 OpenAI 兼容服务，需要通过 extra_body 传递厂商自定义参数
            "extra_body": {
                "enable_thinking": False,
                "max_thought_tokens": 0
            }
        }
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = requests.post(QWEN_API_URL, headers=headers, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    # 保险起见，去除可能残留的 <think> 块
                    try:
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
                    except Exception:
                        pass
                    return content.strip()
                else:
                    error_msg = f"API调用失败，状态码: {response.status_code}, 响应: {response.text}"
                    if attempt < self.max_retries - 1:
                        logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                        time.sleep(2 ** attempt)  # 指数退避：2^0, 2^1, 2^2秒
                    else:
                        raise Exception(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"网络请求异常: {str(e)}"
                if attempt < self.max_retries - 1:
                    logger.warning(f"第{attempt+1}次尝试失败，{error_msg}，正在重试...")
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(error_msg)
        
        return ""
    
    async def predict(self, source: str, pair_type: str) -> str:
        """根据source生成预测"""
        try:
            # 构建提示词
            if pair_type == "tool_call":
                user_query = "请根据上下文调用合适的工具。"
            else:
                user_query = "请根据工具返回的结果生成最终回答。"
            
            prompt = [
                {"role": "system", "content": source},
                {"role": "user", "content": user_query}
            ]
            
            # 调用Qwen API
            result = self.call_qwen_api(prompt, temperature=0.0, top_p=1.0)
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
    
    def call_retrieval_tool(self, params: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """调用检索工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": "id",
            "method": "tools/call",
            "params": {
                "name": "retrieval_tool",
                "arguments": params,
            },
        }
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(RETRIEVAL_ENDPOINT, headers=RETRIEVAL_HEADERS, json=payload, timeout=20)
                code = getattr(resp, "status_code", None) or 0
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": resp.text}
                return code, data
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"检索工具调用失败，第{attempt+1}次尝试: {e}")
                    time.sleep(self.retry_delay)
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

    def compute_recall_from_pair1_predict(self, pair1_predict: str, pair2_target: str) -> Tuple[int, Dict[str, Any]]:
        """计算recall指标：基于 pair1 的预测调用中的 query 字段"""
        try:
            params = self.extract_query_params_from_pair1_predict(pair1_predict)
            if not params:
                return 0, {"error": "无法从pair1预测中提取检索参数"}

            status_code, response = self.call_retrieval_tool(params)
            if status_code != 200:
                return 0, {"error": f"检索工具调用失败，状态码: {status_code}"}

            retrieved_tools = self.extract_retrieved_tools(response, top_k=5)

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
    """文本生成评估模块：使用Gemini进行评估"""
    
    def __init__(self, model_type: str = "gemini-2.5-flash"):
        self.model_type = model_type
        self.max_retries = 3
        self.retry_delay = 5
        logger.info(f"初始化文本生成评估模块，使用模型: {model_type}")
    
    def call_gemini_api(self, prompt: str, temperature: float = 0.3, top_p: float = 0.95, top_k: int = 40) -> str:
        """调用Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_type}:generateContent?key={GEMINI_API_KEY}"
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
    
    async def evaluate_text_generation(self, target: str, predict: str) -> Tuple[float, Dict[str, Any]]:
        """使用Gemini评估文本生成质量"""
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
            # 调用Gemini API
            result = self.call_gemini_api(judge_prompt)
            
            # 提取JSON结果
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                eval_result = json.loads(json_match.group())
                overall_score = eval_result.get("overall_score", 0) / 10.0  # 转换为0-1分数
                return overall_score, eval_result
            else:
                return 0.0, {"error": "无法解析评估结果"}
                
        except Exception as e:
            logger.error(f"文本生成评估失败: {e}")
            return 0.0, {"error": str(e)}

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
        self.text_evaluator = TextGenerationEvaluator("gemini-2.5-flash")
        self.retrieval_caller = RetrievalToolCaller()
        self.metrics_calculator = MetricsCalculator()
        logger.info("训练数据评估器初始化完成")
    
    async def evaluate_file(self, file_path: str, checkpoint_file: str = None) -> List[EvaluationResult]:
        """评估整个文件，支持断点续传和实时指标更新"""
        logger.info(f"开始评估文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否有断点文件
        all_results = []
        start_idx = 0
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
        
        # 从断点继续评估
        for idx, conversation_data in enumerate(data[start_idx:], start=start_idx):
            logger.info(f"评估对话 {idx + 1}/{len(data)} (conversation_id: {conversation_id})")
            
            # 解析pairs
            pairs = self.data_processor.parse_conversations(conversation_data, conversation_id)
            
            # 评估每个pair
            pair_predict_by_id = {}
            pair_toolname_score_by_id = {}
            for pair in pairs:
                # 检查是否已处理过该pair
                pair_key = (conversation_id, pair.pair_id)
                if pair_key in processed_pairs:
                    logger.info(f"跳过已处理的 Pair {pair.pair_id}")
                    continue
                
                logger.info(f"评估 Pair {pair.pair_id} (类型: {pair.pair_type})")
                
                # 生成预测
                predict = await self.llm_predictor.predict(pair.source, pair.pair_type)
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
                    
                    # 只有pair2才计算recall指标（且当pair1 precision@1==1.0 且有预测query时才计算）
                    if pair.pair_id == 2:
                        pair1_predict = pair_predict_by_id.get(1)
                        pair1_toolname_score = pair_toolname_score_by_id.get(1)
                        if pair1_predict and pair1_toolname_score == 1.0:
                            recall, recall_details = self.retrieval_caller.compute_recall_from_pair1_predict(pair1_predict, pair.target)
                else:
                    # pair3（文本生成）不涉及recall
                    score, details = await self.text_evaluator.evaluate_text_generation(pair.target, predict)
                    tool_name_score = 0.0
                    recall = None
                    recall_details = None
                
                result = EvaluationResult(
                    conversation_id=conversation_id,
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
                
                all_results.append(result)
                processed_pairs.add(pair_key)
                
                # 根据类型输出不同的日志信息（与日志字段命名保持一致）
                if pair.pair_type == "tool_call":
                    if recall is not None:
                        logger.info(f"Pair {pair.pair_id} 评估完成，accuracy: {score:.3f}, precision@1: {tool_name_score:.3f}, recall@5: {recall}")
                    else:
                        logger.info(f"Pair {pair.pair_id} 评估完成，accuracy: {score:.3f}, precision@1: {tool_name_score:.3f}")
                else:
                    logger.info(f"Pair {pair.pair_id} 评估完成，answer_score: {score:.3f}")
                
                # 每处理一个评估对就更新实时指标并保存
                realtime_metrics = self.metrics_calculator.update_realtime_metrics(realtime_metrics, all_results)
                self._save_realtime_metrics(realtime_metrics)
                
                # 每处理一个评估对就保存一次断点
                if checkpoint_file:
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
                        "next_conversation_id": conversation_id
                    }
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_data, f, ensure_ascii=False)
            
            conversation_id += 1
        
        return all_results
    
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
                "overall_metrics": overall_metrics
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

async def main():
    """主函数"""
    logger.add("metrics/data_evaluation.log", rotation="10 MB")
    
    # 创建输出目录
    os.makedirs("metrics", exist_ok=True)
    
    # 配置参数
    demo_file = "data/9.17_evaluate_data_top5_final.json"
    output_file = "metrics/data_evaluation_results.json"
    checkpoint_file = "metrics/evaluation_checkpoint.json"
    realtime_file = "metrics/realtime_metrics.json"
    
    logger.info("开始增强版训练数据评估")
    
    # 初始化评估器
    evaluator = TrainingDataEvaluator()
    
    # 执行评估，支持断点续传和实时指标更新
    results = await evaluator.evaluate_file(demo_file, checkpoint_file)
    
    # 生成最终报告
    report = evaluator.generate_report(results)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(_round_floats(report, 3), f, ensure_ascii=False, indent=2)
    
    # 评估完成后删除断点文件
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            logger.info(f"已删除断点文件: {checkpoint_file}")
        except Exception as e:
            logger.error(f"删除断点文件失败: {e}")
    
    

if __name__ == "__main__":
    asyncio.run(main())