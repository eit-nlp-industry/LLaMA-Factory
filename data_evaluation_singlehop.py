#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单跳模型评估脚本（旧模板版）
--------------------------------------------------
1. 评估对象：基于旧版单跳template的工具调用数据（单次human→function_call）。
2. Prompt格式：沿用旧模板（system+user+assistant占位，无多跳上下文拼接）。
3. 服务依赖：
   - 推理：SINGLEHOP_VLLM_BASE_URL（默认 http://localhost:5526，对应端口 5526）
   - 检索：SINGLEHOP_RETRIEVAL_ENDPOINT（可选，默认 http://127.0.0.1:9527）
   - 可通过环境变量覆盖，脚本会在启动时打印端口汇总，便于排查。
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

# =========================
# 服务端口 & 环境变量定义
# =========================
SINGLEHOP_VLLM_BASE_URL = os.getenv("SINGLEHOP_VLLM_BASE_URL", "http://localhost:5526")
SINGLEHOP_API_KEY = os.getenv("SINGLEHOP_API_KEY", "")
SINGLEHOP_MODEL_NAME = os.getenv("SINGLEHOP_MODEL_NAME", "my_lora")
SINGLEHOP_API_URL = f"{SINGLEHOP_VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"

# 检索工具服务（用于Recall评估）
SINGLEHOP_RETRIEVAL_ENDPOINT = os.getenv("SINGLEHOP_RETRIEVAL_ENDPOINT", "http://127.0.0.1:9527/v1/databoard/tools/call")
SINGLEHOP_RETRIEVAL_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
DISABLE_RECALL = str(os.getenv("SINGLEHOP_DISABLE_RECALL", "0")).lower() in ("1", "true", "yes")

# 统一汇总脚本涉及到的服务端口，方便排查
SERVICE_PORTS = {
    "vllm_inference": SINGLEHOP_VLLM_BASE_URL,
}
if not DISABLE_RECALL and SINGLEHOP_RETRIEVAL_ENDPOINT:
    SERVICE_PORTS["retrieval_tool"] = SINGLEHOP_RETRIEVAL_ENDPOINT

# 并发配置（旧模板数据普遍较小，默认串行，可通过环境变量调整）
MAX_CONCURRENT_EXAMPLES = int(os.getenv("SINGLEHOP_MAX_CONCURRENT", "5"))

# 业务跳评估配置：是否使用训练数据中的observation（而非实际检索结果）
# 设置为True时，业务跳评估使用训练数据中的observation，消除分布不匹配问题
# 适合测试过拟合效果和模型对训练数据的记忆程度
USE_TRAINING_DATA_OBSERVATION = os.getenv("SINGLEHOP_USE_TRAINING_OBSERVATION", "false").lower() in ("true", "1", "yes")


@dataclass
class SingleHopExample:
    conversation_id: int
    pair_id: int
    system_prompt: str
    user_prompt: str
    target: str  # 期望的工具调用
    next_tool_name: str = ""  # 下一跳业务工具，用于recall
    tools: str = ""  # 工具定义（JSON字符串），用于评估时让模型看到工具列表
    is_retrieval_hop: bool = False  # 是否为检索跳
    related_retrieval_pair_id: int = 0  # 如果是业务跳，关联的检索跳pair_id


@dataclass
class SingleHopResult:
    conversation_id: int
    pair_id: int
    predict: str
    target: str
    score: float
    tool_name_score: float
    details: Dict[str, Any]
    recall: Optional[int] = None
    recall_details: Optional[Dict[str, Any]] = None
    arg_match: Optional[int] = None  # 严格模式下的参数匹配：1=完全匹配，0=不匹配，None=跳过评估


class ToolCallEvaluator:
    """基础的工具调用评估（与多跳版保持一致简化版）"""

    IGNORED_FIELDS = {"user_id", "trace_id", "top_k"}

    def __init__(self, strict_mode: bool = False):
        """
        初始化评估器
        
        Args:
            strict_mode: 是否启用严格模式
                - False (默认): 原有逻辑，工具名匹配0.5分 + 参数匹配0.5分（按比例）
                - True: 严格模式，只有在工具名匹配时才计算参数，参数必须完全匹配才得分
        """
        self.strict_mode = strict_mode

    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {}
        try:
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)
            match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
        return {}

    def evaluate(self, target: str, predict: str) -> tuple[float, float, Dict[str, Any], Optional[int]]:
        """
        评估工具调用
        
        Returns:
            (score, tool_name_score, details, arg_match)
            - score: 综合得分（原有逻辑）
            - tool_name_score: 工具名匹配得分（1.0或0.0）
            - details: 详细评估信息
            - arg_match: 严格模式下的参数匹配（1=完全匹配，0=不匹配，None=跳过评估）
        """
        target_call = self.extract_tool_call(target)
        predict_call = self.extract_tool_call(predict)

        details = {
            "target_call": target_call,
            "predict_call": predict_call,
            "tool_name_match": False,
            "arguments_match": False,
            "argument_details": {},
            "raw_predict": predict[:500],  # 保存原始预测用于调试
            "raw_target": target[:500],     # 保存原始目标用于调试
        }

        if not target_call:
            logger.warning("目标工具调用解析失败，target预览: {}", target[:200])
            logger.debug("完整target内容: {}", target)
            return 0.0, 0.0, details, None

        if not predict_call:
            logger.warning("预测工具调用解析失败，predict预览: {}", predict[:200])
            logger.debug("完整predict内容: {}", predict)
            return 0.0, 0.0, details, None
        
        # 记录解析成功的工具调用信息
        logger.debug("目标工具: {}, 预测工具: {}", target_call.get("name", "未知"), predict_call.get("name", "未知"))

        score = 0.0
        tool_name_score = 0.0
        arg_match = None  # 严格模式下的参数匹配结果

        target_name = target_call.get("name", "")
        predict_name = predict_call.get("name", "")
        tool_name_matched = target_name and target_name == predict_name
        
        if tool_name_matched:
            details["tool_name_match"] = True
            tool_name_score = 1.0
            score += 0.5

        target_args = target_call.get("arguments", {}) or {}
        predict_args = predict_call.get("arguments", {}) or {}
        filtered_target_args = {k: v for k, v in target_args.items() if k not in self.IGNORED_FIELDS}

        # 严格模式：只有在工具名匹配时才计算参数匹配
        if self.strict_mode:
            if tool_name_matched and filtered_target_args:
                # 工具名匹配且有参数，计算参数匹配
                all_match = True
                for key, value in filtered_target_args.items():
                    predict_value = predict_args.get(key)
                    is_match = (predict_value == value)
                    details["argument_details"][key] = {
                        "target": value,
                        "predict": predict_value,
                        "match": is_match,
                    }
                    if not is_match:
                        all_match = False
                arg_match = 1 if all_match else 0
                details["arguments_match"] = all_match
                # 严格模式下，参数必须完全匹配才得分
                if all_match:
                    score += 0.5
            elif tool_name_matched and target_args and not filtered_target_args:
                # 工具名匹配但仅剩忽略字段
                arg_match = 1
                details["arguments_match"] = True
                score += 0.5
            else:
                # 工具名不匹配，跳过参数评估
                arg_match = None
                details["arguments_match"] = False
        else:
            # 原有逻辑：即使工具名不匹配也计算参数（虽然总分已经是0了）
            if filtered_target_args:
                matches = 0
                for key, value in filtered_target_args.items():
                    predict_value = predict_args.get(key)
                    details["argument_details"][key] = {
                        "target": value,
                        "predict": predict_value,
                        "match": predict_value == value,
                    }
                    if predict_value == value:
                        matches += 1
                score += 0.5 * (matches / len(filtered_target_args))
                details["arguments_match"] = matches == len(filtered_target_args)
            elif target_args:
                # 仅剩忽略字段
                score += 0.5
                details["arguments_match"] = True

        return score, tool_name_score, details, arg_match


class SingleHopDataProcessor:
    """从旧模板数据中抽取单跳评估样本"""

    def parse(self, data: List[Dict[str, Any]], diagnostic_mode: bool = False, eval_all_hops: bool = True) -> List[SingleHopExample]:
        """
        解析数据，提取所有工具调用评估样本
        
        Args:
            data: 对话数据列表
            diagnostic_mode: 是否启用诊断模式
            eval_all_hops: 是否评估所有跳（True: 评估检索跳+业务跳, False: 仅评估检索跳）
        """
        examples: List[SingleHopExample] = []
        pair_id = 1

        for idx, conversation in enumerate(data, start=1):
            system_prompt = conversation.get("system", "")
            tools = conversation.get("tools", "[]")  # 提取tools字段
            conv = conversation.get("conversations", [])
            
            # 用于累积对话上下文（用于业务工具调用评估）
            conversation_context = []
            
            for i in range(len(conv) - 1):
                msg = conv[i]
                nxt = conv[i + 1]
                
                # 评估检索跳：human -> function_call (通常是 retrieval_tool)
                if msg.get("from") == "human" and nxt.get("from") == "function_call":
                    prompt = self._build_prompt(system_prompt, msg["value"], tools)
                    next_tool_name = self._find_next_tool_name(conv, i + 1)
                    example = SingleHopExample(
                        conversation_id=idx,
                        pair_id=pair_id,
                        system_prompt=prompt["system"],
                        user_prompt=prompt["user"],
                        target=nxt["value"],
                        next_tool_name=next_tool_name,
                        tools=tools,  # 保存tools字段
                    )
                    examples.append(example)
                    
                    # 诊断模式：显示前5个样本的详细信息
                    if diagnostic_mode and pair_id <= 5:
                        self._log_example_details(example, pair_id)
                    
                    # 累积上下文（用于后续业务工具调用）
                    conversation_context.append(msg)  # human
                    conversation_context.append(nxt)  # retrieval_tool call
                    if i + 2 < len(conv):
                        conversation_context.append(conv[i + 2])  # observation
                    
                    pair_id += 1
                    if not eval_all_hops:
                        break  # 旧行为：只评估检索跳
                
                # 评估业务跳：observation -> function_call (业务工具，如 list_orders)
                elif eval_all_hops and msg.get("from") == "observation" and nxt.get("from") == "function_call":
                    # 找到关联的检索跳pair_id（上一个检索跳）
                    related_retrieval_pair_id = 0
                    for prev_ex in reversed(examples):
                        if prev_ex.conversation_id == idx and prev_ex.is_retrieval_hop:
                            related_retrieval_pair_id = prev_ex.pair_id
                            break
                    
                    # 构建包含历史对话的prompt（注意：这里先使用训练数据中的observation，后面评估时会替换为实际检索结果）
                    user_prompt = self._build_user_prompt_with_context(conversation_context, msg, tools)
                    prompt = self._build_prompt(system_prompt, user_prompt, tools)
                    next_tool_name = self._find_next_tool_name(conv, i + 1)
                    example = SingleHopExample(
                        conversation_id=idx,
                        pair_id=pair_id,
                        system_prompt=prompt["system"],
                        user_prompt=prompt["user"],
                        target=nxt["value"],
                        next_tool_name=next_tool_name,
                        tools=tools,
                        is_retrieval_hop=False,
                        related_retrieval_pair_id=related_retrieval_pair_id,  # 关联的检索跳pair_id
                    )
                    examples.append(example)
                    
                    # 诊断模式：显示前5个样本的详细信息
                    if diagnostic_mode and pair_id <= 5:
                        self._log_example_details(example, pair_id)
                    
                    # 继续累积上下文
                    conversation_context.append(msg)  # observation
                    conversation_context.append(nxt)  # business_tool call
                    if i + 2 < len(conv):
                        conversation_context.append(conv[i + 2])  # observation (结果)
                    
                    pair_id += 1

        hop_type_summary = "所有跳" if eval_all_hops else "检索跳"
        logger.info("共解析出 {} 个样本（评估{}）", len(examples), hop_type_summary)
        
        # 统计工具类型
        retrieval_count = 0
        business_count = 0
        for ex in examples:
            try:
                target_obj = json.loads(ex.target) if isinstance(ex.target, str) else ex.target
                tool_name = target_obj.get("name", "") if isinstance(target_obj, dict) else ""
                if tool_name == "retrieval_tool":
                    retrieval_count += 1
                else:
                    business_count += 1
            except:
                pass
        logger.info("  - 检索工具调用: {} 个", retrieval_count)
        logger.info("  - 业务工具调用: {} 个", business_count)
        
        return examples
    
    @staticmethod
    def _log_example_details(example: SingleHopExample, pair_id: int):
        """记录样本的详细信息用于诊断"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("📋 样本 {} 详细信息 (conversation_id: {}, pair_id: {})", pair_id, example.conversation_id, example.pair_id)
        logger.info("=" * 80)
        
        # 解析目标工具调用
        try:
            target_obj = json.loads(example.target)
            target_name = target_obj.get("name", "未知")
            target_args = target_obj.get("arguments", {})
        except:
            target_name = "解析失败"
            target_args = {}
        
        logger.info("🎯 目标工具调用:")
        logger.info("   工具名: {}", target_name)
        logger.info("   参数: {}", json.dumps(target_args, ensure_ascii=False, indent=6))
        logger.info("   原始内容: {}", example.target[:200] + ("..." if len(example.target) > 200 else ""))
        
        logger.info("")
        logger.info("💬 System Prompt:")
        logger.info("   {}", example.system_prompt[:300] + ("..." if len(example.system_prompt) > 300 else ""))
        
        logger.info("")
        logger.info("👤 User Prompt:")
        logger.info("   {}", example.user_prompt[:300] + ("..." if len(example.user_prompt) > 300 else ""))
        
        logger.info("")
        logger.info("🔗 Next Tool Name (用于Recall计算): {}", example.next_tool_name if example.next_tool_name else "无")
        
        logger.info("")
        logger.info("🛠️ Tools字段 (原始):")
        if example.tools:
            try:
                tools_preview = example.tools[:300] + ("..." if len(example.tools) > 300 else "")
                logger.info("   {}", tools_preview)
                # 尝试解析并显示工具数量
                tools_list = json.loads(example.tools) if isinstance(example.tools, str) else example.tools
                if isinstance(tools_list, list):
                    logger.info("   工具数量: {}", len(tools_list))
                    if len(tools_list) > 0:
                        first_tool = tools_list[0]
                        tool_name = first_tool.get("name", "") if isinstance(first_tool, dict) else ""
                        logger.info("   第一个工具: {}", tool_name)
            except:
                logger.info("   (无法解析)")
        else:
            logger.info("   (无)")
        
        logger.info("=" * 80)
        logger.info("")

    @staticmethod
    def _build_prompt(system_prompt: str, user_query: str, tools: str = "[]") -> Dict[str, str]:
        """构建prompt，包含system、tools和user"""
        system_block = system_prompt or "You are a helpful assistant."
        
        # 检测是否包含增强规则的关键词，如果没有则自动替换
        if "第一阶段：检索工具调用" not in system_block:
            try:
                # 尝试从tool_calling_setup.py导入增强的system prompt
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                
                from tool_calling_setup import create_enhanced_system_prompt
                enhanced_prompt = create_enhanced_system_prompt()
                logger.info("✅ 检测到system prompt未包含增强规则，已自动替换为增强版（与训练时一致）")
                system_block = enhanced_prompt
            except ImportError as e:
                logger.warning("⚠️ 无法导入增强的system prompt（ImportError），尝试从文件加载: {}", str(e))
                # 备选方案：从文件加载
                try:
                    import os
                    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhanced_system_prompt.txt")
                    if os.path.exists(prompt_path):
                        with open(prompt_path, 'r', encoding='utf-8') as f:
                            system_block = f.read().strip()
                        logger.info("✅ 从文件加载增强的system prompt: {}", prompt_path)
                    else:
                        logger.warning("⚠️ 增强模板文件不存在: {}，使用原始prompt", prompt_path)
                except Exception as file_e:
                    logger.warning("⚠️ 从文件加载失败: {}，使用原始prompt", str(file_e))
            except Exception as e:
                logger.warning("⚠️ 无法加载增强的system prompt，使用原始prompt: {}", str(e))
        else:
            logger.debug("✅ 检测到system prompt已包含增强规则，直接使用")
        
        user_block = user_query.strip()
        
        # 将tools格式化后追加到system prompt，与训练时保持一致
        # 这样评估时模型也能看到工具列表，而不是仅凭记忆预测
        try:
            tools_list = json.loads(tools) if isinstance(tools, str) else tools
            if tools_list and len(tools_list) > 0:
                # 使用与训练时相同的格式（QWEN_TOOL_PROMPT格式）
                # 动态导入避免循环依赖
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.insert(0, current_dir)
                
                from src.llamafactory.data.tool_utils import QwenToolUtils
                tool_formatter = QwenToolUtils()
                tool_text = tool_formatter.tool_formatter(tools_list)
                # 追加到system prompt后面（与template.py的逻辑一致）
                system_block = system_block + tool_text
                logger.debug("✅ 已添加工具列表到system prompt（工具数量: {}）", len(tools_list))
        except Exception as e:
            logger.warning("⚠️ 解析tools字段失败，将不使用工具列表: {}", str(e))
            logger.debug("Tools字段内容: {}", tools[:200] if tools else "空")
        
        return {
            "system": system_block,
            "user": user_block,
        }

    @staticmethod
    def _build_user_prompt_with_context(context: List[Dict[str, Any]], observation: Dict[str, Any], tools: str = "[]") -> str:
        """
        为业务工具调用构建包含历史上下文的user prompt
        
        Args:
            context: 累积的对话历史（human, tool_call, observation等）
            observation: 当前的observation消息（retrieval_tool的返回结果）
            tools: 工具列表
        
        Returns:
            构建的user prompt字符串
        """
        # 构建完整的对话历史
        prompt_parts = []
        
        # 添加初始用户查询（第一个human消息）
        for msg in context:
            if msg.get("from") == "human":
                prompt_parts.append(f"用户: {msg.get('value', '')}")
                break
        
        # 添加工具调用和返回结果
        for i, msg in enumerate(context):
            if msg.get("from") == "function_call":
                try:
                    call_obj = json.loads(msg.get("value", "{}"))
                    tool_name = call_obj.get("name", "")
                    prompt_parts.append(f"调用工具: {tool_name}")
                except:
                    prompt_parts.append("调用工具: [解析失败]")
            elif msg.get("from") == "observation":
                # 截断过长的observation
                obs_value = msg.get("value", "")
                if len(obs_value) > 500:
                    obs_value = obs_value[:500] + "..."
                prompt_parts.append(f"工具返回: {obs_value}")
        
        # 添加当前的observation（retrieval_tool的返回结果）
        obs_value = observation.get("value", "")
        if len(obs_value) > 500:
            obs_value = obs_value[:500] + "..."
        prompt_parts.append(f"工具返回: {obs_value}")
        
        # 组合成完整的prompt
        user_prompt = "\n".join(prompt_parts)
        
        return user_prompt

    @staticmethod
    def _find_next_tool_name(conversations: List[Dict[str, Any]], start_idx: int) -> str:
        """在首个function_call之后查找下一次business_tool调用"""
        for j in range(start_idx + 1, len(conversations)):
            msg = conversations[j]
            if msg.get("from") == "function_call":
                try:
                    call_obj = json.loads(msg.get("value", "{}"))
                    name = call_obj.get("name", "")
                    if name and name != "retrieval_tool":
                        return name
                except Exception:
                    continue
        return ""


class RetrievalToolCaller:
    """用于计算retrieval_tool的Recall@5"""

    def __init__(self):
        self.max_retries = 3

    def _extract_tool_call(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {}
        try:
            if text.startswith("{") and text.endswith("}"):
                return json.loads(text)
            match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except Exception:
            pass
        return {}

    def extract_query(self, predict: str) -> str:
        try:
            call_obj = self._extract_tool_call(predict)
            arguments = call_obj.get("arguments", {}) if isinstance(call_obj, dict) else {}
            return arguments.get("query", "")
        except Exception:
            return ""

    async def call_retrieval_tool(self, session: aiohttp.ClientSession, query: str, user_id: int = 13) -> tuple[int, Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": "singlehop_eval",
            "method": "tools/call",
            "params": {
                "name": "retrieval_tool",
                "arguments": {
                    "query": query,
                    "source_filter": "toollist",
                    "user_id": str(user_id),
                    "top_k": 5,
                    "trace_id": "trace_singlehop_eval",
                },
            },
        }

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    SINGLEHOP_RETRIEVAL_ENDPOINT,
                    headers=SINGLEHOP_RETRIEVAL_HEADERS,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    code = resp.status
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"raw": await resp.text()}
                    return code, data
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    return 0, {"error": str(exc)}
                await asyncio.sleep(2 ** attempt)
        return 0, {"error": "retrieval request failed"}

    @staticmethod
    def build_observation_from_response(response_obj: Dict[str, Any]) -> str:
        """
        从检索服务响应构建observation字符串（用于业务跳评估）
        
        Args:
            response_obj: 检索服务的响应对象
        
        Returns:
            observation字符串（JSON格式）
        """
        try:
            # 尝试提取完整的result作为observation
            result = response_obj.get("result")
            if result:
                # 如果result是字典，尝试提取嵌套的response.result
                if isinstance(result, dict):
                    nested_result = result.get("response", {}).get("result")
                    if nested_result is not None:
                        return json.dumps(nested_result, ensure_ascii=False)
                    # 或者直接使用result
                    return json.dumps(result, ensure_ascii=False)
                # 如果result是列表，直接返回
                elif isinstance(result, list):
                    return json.dumps(result, ensure_ascii=False)
            
            # 如果无法提取，返回整个响应
            return json.dumps(response_obj, ensure_ascii=False)
        except Exception as e:
            logger.warning("构建observation失败: {}, 使用原始响应", str(e))
            return json.dumps(response_obj, ensure_ascii=False)

    @staticmethod
    def extract_tools(response_obj: Dict[str, Any], top_k: int = 5) -> List[str]:
        """
        从检索工具响应中提取工具名称列表
        支持多种响应结构：
        1. {"result": [...]} - 直接列表
        2. {"result": {"response": {"result": [...]}}} - 嵌套结构（与训练数据一致）
        3. {"result": {"tools": [...]}} - 字典中的tools字段
        """
        tools: List[str] = []
        try:
            # 首先尝试从JSON-RPC响应的result字段获取
            result = response_obj.get("result")
            
            # 情况1: result是列表（直接的工具列表）
            if isinstance(result, list):
                iterable = result[:top_k]
            # 情况2: result是字典，可能包含嵌套结构
            elif isinstance(result, dict):
                # 优先检查嵌套结构：result.response.result（与训练数据一致）
                if "response" in result and isinstance(result["response"], dict):
                    nested_result = result["response"].get("result")
                    if isinstance(nested_result, list):
                        iterable = nested_result[:top_k]
                    else:
                        iterable = []
                # 检查result.tools字段
                elif "tools" in result and isinstance(result["tools"], list):
                    iterable = result["tools"][:top_k]
                else:
                    iterable = []
            else:
                iterable = []

            # 从iterable中提取工具名称
            for item in iterable:
                if isinstance(item, dict):
                    # 支持两种响应格式：
                    # 1. 9527端口：result[i].content.name
                    # 2. 8084端口：result[i].name
                    if "content" in item and isinstance(item["content"], dict):
                        tool_name = item["content"].get("name", "")
                        if tool_name:
                            tools.append(tool_name)
                            continue
                    # 直接字段（8084端口格式）
                    if "name" in item and isinstance(item["name"], str):
                        tools.append(item["name"])
                        continue
                    # 其他可能的字段
                    for key in ("tool_name", "title", "id", "label", "api_name"):
                        if key in item and isinstance(item[key], str):
                            tools.append(item[key])
                            break
            
            logger.debug("从检索响应中提取到 {} 个工具: {}", len(tools), tools)
        except Exception as e:
            logger.warning("提取工具列表失败: {}, 响应结构: {}", str(e), json.dumps(response_obj, ensure_ascii=False)[:500])
        return tools[:top_k]

    async def compute_recall(self, session: aiohttp.ClientSession, predict: str, next_tool_name: str) -> tuple[int, Dict[str, Any]]:
        logger.debug("开始计算Recall - next_tool_name: {}, predict预览: {}", next_tool_name, predict[:200])
        query = self.extract_query(predict)
        if not query:
            logger.warning("无法从预测中提取query，predict内容: {}", predict[:300])
            return 0, {"error": "无法从预测中提取query", "predict_preview": predict[:300]}

        logger.info("调用检索服务 - query: {}", query[:100])
        status, response = await self.call_retrieval_tool(session, query)
        if status != 200:
            logger.error("检索服务调用失败，状态码: {}, 响应: {}", status, str(response)[:500])
            return 0, {"error": f"检索服务状态码: {status}", "response": response}

        # 诊断：记录完整响应结构（用于调试）
        logger.debug("检索服务响应结构: {}", json.dumps(response, ensure_ascii=False)[:1000])
        
        retrieved_tools = self.extract_tools(response, top_k=5)
        logger.info("检索返回工具列表 (共{}个): {}", len(retrieved_tools), retrieved_tools)
        recall = 1 if next_tool_name in retrieved_tools else 0
        logger.info("Recall结果: {} (目标工具: {} 是否在检索结果中: {})", recall, next_tool_name, next_tool_name in retrieved_tools)

        details = {
            "query": query,
            "target_tool": next_tool_name,
            "retrieved_tools": retrieved_tools,
            "recall": recall,
            "response_status": status,
        }
        return recall, details


class LLMPredictor:
    def __init__(self):
        self.model = SINGLEHOP_MODEL_NAME
        self.max_retries = 4

    async def infer(self, session: aiohttp.ClientSession, example: SingleHopExample, diagnostic_mode: bool = False) -> str:
        # 使用qwen3 template格式，与训练时保持一致
        # 训练时使用：<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
        # 评估时也应该使用相同格式，确保一致性
        qwen3_prompt = f"<|im_start|>system\n{example.system_prompt}<|im_end|>\n<|im_start|>user\n{example.user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # 提取system和user内容（VLLM API需要role-based格式，但会使用chat_template自动转换）
        # 注意：如果VLLM支持chat_template_kwargs，会自动应用qwen3 template
        prompt = [
            {"role": "system", "content": example.system_prompt},
            {"role": "user", "content": example.user_prompt},
        ]
        
        # 诊断模式：显示发送给模型的 prompt
        if diagnostic_mode and example.pair_id <= 3:
            logger.info("")
            logger.info("=" * 80)
            logger.info("📤 发送给模型的 Prompt (pair_id: {})", example.pair_id)
            logger.info("=" * 80)
            logger.info("Qwen3 Template格式 (训练时使用的格式):")
            logger.info("   {}", qwen3_prompt[:500] + ("..." if len(qwen3_prompt) > 500 else ""))
            logger.info("")
            logger.info("API格式 (role-based):")
            logger.info("   System: {}", example.system_prompt[:300] + ("..." if len(example.system_prompt) > 300 else ""))
            logger.info("   User: {}", example.user_prompt[:300] + ("..." if len(example.user_prompt) > 300 else ""))
            logger.info("")
            logger.info("Tools字段 (原始):")
            logger.info("   {}", example.tools[:500] + ("..." if len(example.tools) > 500 else ""))
            logger.info("=" * 80)
            logger.info("")
        
        payload = {
            "model": self.model,
            "messages": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        headers = {"Content-Type": "application/json"}
        if SINGLEHOP_API_KEY:
            headers["Authorization"] = f"Bearer {SINGLEHOP_API_KEY}"

        for attempt in range(self.max_retries):
            try:
                async with session.post(SINGLEHOP_API_URL, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                    logger.warning("推理失败[{}/{}] status={}, body={}", attempt + 1, self.max_retries, resp.status, await resp.text())
            except Exception as exc:
                logger.warning("推理异常[{}/{}]: {}", attempt + 1, self.max_retries, exc)
            await asyncio.sleep(2 ** attempt)

        raise RuntimeError(f"推理多次失败 (pair_id={example.pair_id})")


class SingleHopEvaluator:
    def __init__(self, strict_mode: bool = False):
        """
        初始化评估器
        
        Args:
            strict_mode: 是否启用严格模式（参数评估仅在工具名匹配时进行）
        """
        self.data_processor = SingleHopDataProcessor()
        self.predictor = LLMPredictor()
        self.evaluator = ToolCallEvaluator(strict_mode=strict_mode)
        self.retrieval_caller = None if DISABLE_RECALL else RetrievalToolCaller()
        self.strict_mode = strict_mode

    async def evaluate_file(self, input_file: str, diagnostic_mode: bool = False, eval_all_hops: bool = True) -> List[SingleHopResult]:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = self.data_processor.parse(data, diagnostic_mode=diagnostic_mode, eval_all_hops=eval_all_hops)
        if not examples:
            return []

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_EXAMPLES)
        timeout = aiohttp.ClientTimeout(total=120)
        results: List[SingleHopResult] = []

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXAMPLES)

            async def run_example(example: SingleHopExample, diagnostic_mode: bool = False):
                async with semaphore:
                    logger.debug("评估 Pair {} (conversation_id: {})", example.pair_id, example.conversation_id)
                    predict = await self.predictor.infer(session, example, diagnostic_mode=diagnostic_mode)
                    logger.debug("Pair {} 预测结果长度: {}, 预览: {}", example.pair_id, len(predict), predict[:200])
                    
                    # 诊断模式：显示预测结果详情
                    if diagnostic_mode and example.pair_id <= 3:
                        logger.info("")
                        logger.info("=" * 80)
                        logger.info("📥 模型返回结果 (pair_id: {})", example.pair_id)
                        logger.info("=" * 80)
                        logger.info("完整预测内容:")
                        logger.info("   {}", predict)
                        logger.info("=" * 80)
                        logger.info("")
                    
                    score, name_score, details, arg_match = self.evaluator.evaluate(example.target, predict)
                    
                    # 记录评估结果
                    target_name = details.get("target_call", {}).get("name", "未知")
                    predict_name = details.get("predict_call", {}).get("name", "未知")
                    logger.info("Pair {} 评估结果: score={:.3f}, name_score={:.3f}, 目标工具={}, 预测工具={}", 
                               example.pair_id, score, name_score, target_name, predict_name)
                    if self.strict_mode and arg_match is not None:
                        logger.debug("Pair {} 参数匹配 (严格模式): {}", example.pair_id, "✅ 完全匹配" if arg_match == 1 else "❌ 不匹配")
                    elif self.strict_mode:
                        logger.debug("Pair {} 参数评估跳过 (严格模式): 工具名不匹配", example.pair_id)
                    
                    recall = None
                    recall_details = None

                    target_name = details.get("target_call", {}).get("name")
                    recall_conditions = {
                        "retrieval_caller_exists": self.retrieval_caller is not None,
                        "target_is_retrieval": target_name == "retrieval_tool",
                        "has_next_tool": bool(example.next_tool_name),
                        "name_score_is_1": name_score == 1.0
                    }
                    
                    logger.debug("Pair {} Recall计算条件: {}", example.pair_id, recall_conditions)
                    
                    if (
                        self.retrieval_caller
                        and target_name == "retrieval_tool"
                        and example.next_tool_name
                        and name_score == 1.0
                    ):
                        logger.info("Pair {} 满足Recall计算条件，开始计算Recall", example.pair_id)
                        try:
                            # 同时使用真实query和预测query计算Recall
                            # 1. 从target（真实）中提取query，用于评估检索服务本身的召回率
                            target_call = details.get("target_call", {})
                            target_query = target_call.get("arguments", {}).get("query", "") if isinstance(target_call.get("arguments"), dict) else ""
                            
                            # 2. 从predict中提取query（用于评估模型预测query的质量）
                            predict_call = details.get("predict_call", {})
                            predict_query = predict_call.get("arguments", {}).get("query", "") if isinstance(predict_call.get("arguments"), dict) else ""
                            
                            # 3. 优先使用真实query计算Recall（评估检索服务本身的质量）
                            # 这样即使模型预测的query有差异，也能正确评估检索服务
                            recall = None
                            recall_details = None
                            
                            if target_query:
                                # 使用真实query计算Recall（这是主要指标）
                                recall, recall_details = await self.retrieval_caller.compute_recall(
                                    session, example.target, example.next_tool_name
                                )
                                logger.info("Pair {} Recall(使用真实query): {}", example.pair_id, recall)
                                
                                # 如果真实query的Recall也低，说明检索服务本身召回率有问题
                                if recall == 0:
                                    logger.warning("Pair {} Recall=0 (使用真实query)，说明检索服务召回率可能有问题", example.pair_id)
                            else:
                                logger.warning("Pair {} 无法从target提取query，使用预测query计算Recall", example.pair_id)
                                # 如果真实query提取失败，回退到使用预测query
                                recall, recall_details = await self.retrieval_caller.compute_recall(
                                    session, predict, example.next_tool_name
                                )
                            
                            # 4. 可选：如果真实query和预测query不同，也计算预测query的Recall（用于对比）
                            if target_query and predict_query and target_query != predict_query and recall is not None:
                                recall_with_predict_query, recall_details_predict = await self.retrieval_caller.compute_recall(
                                    session, predict, example.next_tool_name
                                )
                                logger.info("Pair {} Recall(使用预测query): {} (对比用)", example.pair_id, recall_with_predict_query)
                                
                                # 记录对比信息到recall_details
                                if recall_details:
                                    recall_details["recall_with_predict_query"] = recall_with_predict_query
                                    recall_details["target_query"] = target_query
                                    recall_details["predict_query"] = predict_query
                                    recall_details["query_match"] = False
                                    
                                    if recall == 1 and recall_with_predict_query == 0:
                                        logger.warning("Pair {} Query差异导致Recall下降: 真实query Recall=1, 预测query Recall=0", example.pair_id)
                            elif recall_details:
                                # query相同，记录匹配信息
                                recall_details["query_match"] = target_query == predict_query if target_query else False
                                recall_details["target_query"] = target_query
                                recall_details["predict_query"] = predict_query
                                
                        except Exception as exc:
                            logger.error("Recall计算失败 (pair_id=%s): %s", example.pair_id, exc)
                    else:
                        logger.debug("Pair {} 不满足Recall计算条件: {}", example.pair_id, recall_conditions)

                    results.append(
                        SingleHopResult(
                            conversation_id=example.conversation_id,
                            pair_id=example.pair_id,
                            predict=predict,
                            target=example.target,
                            score=score,
                            tool_name_score=name_score,
                            details=details,
                            recall=recall,
                            recall_details=recall_details,
                            arg_match=arg_match,
                        )
                    )

            # 分两阶段评估（检索跳先评估，业务跳使用实际检索结果）
            # 阶段1：先评估所有检索跳，保存预测结果和实际检索响应
            retrieval_examples_list = [ex for ex in examples if ex.is_retrieval_hop]
            business_examples_list = [ex for ex in examples if not ex.is_retrieval_hop]
            
            if retrieval_examples_list:
                logger.info("")
                logger.info("=" * 80)
                logger.info("阶段1: 评估检索跳（共 {} 个）", len(retrieval_examples_list))
                logger.info("=" * 80)
                retrieval_cache: Dict[tuple[int, int], tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]] = {}
                
                async def run_retrieval_with_cache(ex: SingleHopExample):
                    async with semaphore:
                        logger.debug("评估检索跳 Pair {} (conversation_id: {})", ex.pair_id, ex.conversation_id)
                        predict = await self.predictor.infer(session, ex, diagnostic_mode=diagnostic_mode)
                        score, name_score, details, arg_match = self.evaluator.evaluate(ex.target, predict)
                        
                        target_name = details.get("target_call", {}).get("name")
                        actual_retrieval_response = None
                        recall = None
                        recall_details = None
                        
                        # 如果是retrieval_tool且预测成功，获取实际检索响应
                        if (target_name == "retrieval_tool" and name_score == 1.0 and 
                            self.retrieval_caller and ex.next_tool_name):
                            try:
                                predict_call = details.get("predict_call", {})
                                predict_query = predict_call.get("arguments", {}).get("query", "") if isinstance(predict_call.get("arguments"), dict) else ""
                                if predict_query:
                                    status, retrieval_response = await self.retrieval_caller.call_retrieval_tool(session, predict_query)
                                    if status == 200:
                                        actual_retrieval_response = retrieval_response
                                        
                                # 计算Recall（使用真实query）
                                target_call = details.get("target_call", {})
                                target_query = target_call.get("arguments", {}).get("query", "") if isinstance(target_call.get("arguments"), dict) else ""
                                if target_query:
                                    recall, recall_details = await self.retrieval_caller.compute_recall(
                                        session, ex.target, ex.next_tool_name
                                    )
                            except Exception as exc:
                                logger.error("检索或Recall计算失败: {}", exc)
                        
                        # 保存结果
                        results.append(SingleHopResult(
                            conversation_id=ex.conversation_id, pair_id=ex.pair_id,
                            predict=predict, target=ex.target, score=score,
                            tool_name_score=name_score, details=details,
                            recall=recall, recall_details=recall_details,
                            arg_match=arg_match,
                        ))
                        
                        # 返回这个检索跳的缓存条目（单个条目）
                        return {(ex.conversation_id, ex.pair_id): (predict, actual_retrieval_response or {}, recall_details)}
                
                # 并发评估检索跳并合并缓存
                cache_results_list = await asyncio.gather(*(run_retrieval_with_cache(ex) for ex in retrieval_examples_list))
                final_retrieval_cache = {}
                for cache in cache_results_list:
                    if cache:
                        final_retrieval_cache.update(cache)
                logger.info("✅ 检索跳评估完成，已保存 {} 个检索结果", len(final_retrieval_cache))
                
                # 阶段2：评估业务跳，使用检索跳的实际检索结果
                if business_examples_list:
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info("阶段2: 评估业务跳（共 {} 个，使用实际检索结果）", len(business_examples_list))
                    logger.info("=" * 80)
                    
                    async def run_business_with_actual_retrieval(ex: SingleHopExample):
                        async with semaphore:
                            logger.debug("评估业务跳 Pair {} (conversation_id: {}, 关联检索跳: {})", 
                                       ex.pair_id, ex.conversation_id, ex.related_retrieval_pair_id)
                            
                            # 获取关联检索跳的实际检索结果
                            actual_observation = None
                            used_actual_retrieval = False
                            
                            # 如果配置为使用训练数据observation，直接跳过实际检索
                            if USE_TRAINING_DATA_OBSERVATION:
                                logger.info("业务跳 Pair {} 使用训练数据中的observation（配置启用：USE_TRAINING_DATA_OBSERVATION=True）", ex.pair_id)
                            elif ex.related_retrieval_pair_id > 0:
                                cache_key = (ex.conversation_id, ex.related_retrieval_pair_id)
                                if cache_key in final_retrieval_cache:
                                    _, retrieval_response, _ = final_retrieval_cache[cache_key]
                                    if retrieval_response and (retrieval_response.get("result") or not retrieval_response.get("error")):
                                        if self.retrieval_caller:
                                            actual_observation = self.retrieval_caller.build_observation_from_response(retrieval_response)
                                            if actual_observation:
                                                used_actual_retrieval = True
                                                logger.info("业务跳 Pair {} 使用实际检索结果（来自检索跳 Pair {}）", 
                                                          ex.pair_id, ex.related_retrieval_pair_id)
                                    else:
                                        logger.warning("业务跳 Pair {} 关联的检索跳 {} 检索响应为空或失败，使用训练数据中的observation", 
                                                     ex.pair_id, ex.related_retrieval_pair_id)
                                else:
                                    logger.warning("业务跳 Pair {} 找不到关联的检索跳 {} 的缓存，使用训练数据中的observation", 
                                                 ex.pair_id, ex.related_retrieval_pair_id)
                            
                            # 如果有实际检索结果且未配置使用训练数据，重新构建prompt
                            if actual_observation and used_actual_retrieval and not USE_TRAINING_DATA_OBSERVATION:
                                # 替换user_prompt中的observation
                                # 查找最后一个"工具返回:"并替换其后的内容
                                pattern = r"工具返回:.*$"
                                # 限制observation长度，避免prompt过长
                                obs_preview = actual_observation[:1000] + ("..." if len(actual_observation) > 1000 else "")
                                new_user_prompt = re.sub(pattern, f"工具返回: {obs_preview}", ex.user_prompt, count=1, flags=re.MULTILINE | re.DOTALL)
                                prompt = self.data_processor._build_prompt(ex.system_prompt, new_user_prompt, ex.tools)
                                ex = SingleHopExample(
                                    conversation_id=ex.conversation_id, pair_id=ex.pair_id,
                                    system_prompt=prompt["system"], user_prompt=prompt["user"],
                                    target=ex.target, next_tool_name=ex.next_tool_name, tools=ex.tools,
                                    is_retrieval_hop=False, related_retrieval_pair_id=ex.related_retrieval_pair_id,
                                )
                            else:
                                logger.info("业务跳 Pair {} 使用训练数据中的observation（无实际检索结果或检索失败）", ex.pair_id)
                            
                            # 评估业务跳
                            predict = await self.predictor.infer(session, ex, diagnostic_mode=diagnostic_mode)
                            score, name_score, details, arg_match = self.evaluator.evaluate(ex.target, predict)
                            
                            results.append(SingleHopResult(
                                conversation_id=ex.conversation_id, pair_id=ex.pair_id,
                                predict=predict, target=ex.target, score=score,
                                tool_name_score=name_score, details=details,
                                recall=None, recall_details=None,
                                arg_match=arg_match,
                            ))
                    
                    # 并发评估业务跳
                    await asyncio.gather(*(run_business_with_actual_retrieval(ex) for ex in business_examples_list))
                    logger.info("✅ 业务跳评估完成")
            else:
                # 如果没有检索跳，按原逻辑处理
                await asyncio.gather(*(run_example(ex, diagnostic_mode=diagnostic_mode) for ex in examples))

        return results

    @staticmethod
    def summarize(results: List[SingleHopResult]) -> Dict[str, Any]:
        if not results:
            return {"total": 0, "accuracy": 0.0, "precision@1": 0.0, "recall@5": None, "arg_accuracy": None, "arg_denominator": 0}

        total = len(results)
        acc = sum(r.score for r in results) / total
        p_at1 = sum(r.tool_name_score for r in results) / total
        recall_samples = [r for r in results if r.recall is not None]
        recall_rate = (
            sum(r.recall for r in recall_samples) / len(recall_samples) if recall_samples else None
        )
        
        # 计算参数准确率（严格模式）：仅在 arg_match 不为 None 时统计
        arg_eligible = [r for r in results if r.arg_match is not None]
        arg_hits = sum(1 for r in arg_eligible if r.arg_match == 1)
        arg_accuracy = (arg_hits / len(arg_eligible)) if arg_eligible else None
        
        # 添加详细的失败原因统计
        failure_stats = {
            "parse_failures": {
                "target_parse_failed": 0,
                "predict_parse_failed": 0,
                "both_parse_failed": 0
            },
            "tool_name_mismatches": 0,
            "argument_mismatches": 0,
            "perfect_matches": 0
        }
        
        for r in results:
            details = r.details
            target_call = details.get("target_call", {})
            predict_call = details.get("predict_call", {})
            
            if not target_call and not predict_call:
                failure_stats["parse_failures"]["both_parse_failed"] += 1
            elif not target_call:
                failure_stats["parse_failures"]["target_parse_failed"] += 1
            elif not predict_call:
                failure_stats["parse_failures"]["predict_parse_failed"] += 1
            elif target_call.get("name") != predict_call.get("name"):
                failure_stats["tool_name_mismatches"] += 1
            elif not details.get("arguments_match", False):
                failure_stats["argument_mismatches"] += 1
            else:
                failure_stats["perfect_matches"] += 1
        
        # 分别统计检索跳和业务跳的性能
        retrieval_results = []
        business_results = []
        for r in results:
            try:
                target_obj = json.loads(r.target) if isinstance(r.target, str) else r.target
                tool_name = target_obj.get("name", "") if isinstance(target_obj, dict) else ""
                if tool_name == "retrieval_tool":
                    retrieval_results.append(r)
                else:
                    business_results.append(r)
            except:
                pass
        
        retrieval_stats = {
            "count": len(retrieval_results),
            "accuracy": sum(r.score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0,
            "precision@1": sum(r.tool_name_score for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0,
        }
        
        business_stats = {
            "count": len(business_results),
            "accuracy": sum(r.score for r in business_results) / len(business_results) if business_results else 0.0,
            "precision@1": sum(r.tool_name_score for r in business_results) / len(business_results) if business_results else 0.0,
        }
        
        # 分析Recall失败的原因
        recall_failure_analysis = {
            "total_calculated": len(recall_samples),
            "recall_hits": sum(r.recall for r in recall_samples),
            "recall_misses": len(recall_samples) - sum(r.recall for r in recall_samples),
            "query_extraction_failures": 0,
            "retrieval_service_failures": 0,
            "target_tool_not_in_results": 0,
        }
        
        for r in recall_samples:
            if r.recall_details:
                details = r.recall_details
                if "error" in details:
                    error_msg = str(details.get("error", ""))
                    if "无法提取query" in error_msg:
                        recall_failure_analysis["query_extraction_failures"] += 1
                    else:
                        recall_failure_analysis["retrieval_service_failures"] += 1
                elif r.recall == 0 and details.get("retrieved_tools"):
                    recall_failure_analysis["target_tool_not_in_results"] += 1
        
        return {
            "total": total, 
            "accuracy": acc, 
            "precision@1": p_at1, 
            "recall@5": recall_rate,
            "arg_accuracy": arg_accuracy,
            "arg_denominator": len(arg_eligible),
            "recall_failure_analysis": recall_failure_analysis,
            "failure_stats": failure_stats,
            "retrieval_stats": retrieval_stats,
            "business_stats": business_stats,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="旧模板单跳模型评估脚本")
    parser.add_argument("--input_file", "-i", type=str, default="data/dataset/12_08/train.json", help="输入数据文件")
    parser.add_argument("--output_file", "-o", type=str, default="data/dataset/result/12_15_singlehop_train_eval_results.json", help="评估结果输出文件")
    parser.add_argument("--log_file", "-l", type=str, default="data/dataset/log/singlehop_eval.log", help="日志输出文件")
    parser.add_argument("--diagnostic", "-d", action="store_true", help="启用诊断模式，显示详细的样本和预测信息")
    parser.add_argument("--eval_all_hops", action="store_true", default=True, help="评估所有跳（检索跳+业务跳），默认True")
    parser.add_argument("--eval_retrieval_only", action="store_true", help="仅评估检索跳（与--eval_all_hops互斥）")
    parser.add_argument("--strict_mode", action="store_true", help="启用严格模式：参数评估仅在工具名匹配时进行，参数必须完全匹配才得分（类似eval_via_prod_sse.py）")
    return parser.parse_args()


def dump_report(results: List[SingleHopResult], output_file: str):
    report = {
        "summary": SingleHopEvaluator.summarize(results),
        "cases": [
            {
                "conversation_id": r.conversation_id,
                "pair_id": r.pair_id,
                "score": r.score,
                "tool_name_score": r.tool_name_score,
                "target": r.target,
                "predict": r.predict,
                "details": r.details,
                "recall": r.recall,
                "recall_details": r.recall_details,
                "arg_match": r.arg_match,
            }
            for r in results
        ],
        "service_ports": SERVICE_PORTS,
        "model": SINGLEHOP_MODEL_NAME,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


class SelfCheckValidator:
    """自检验证器：检查脚本配置和数据格式"""
    
    @staticmethod
    async def check_api_service(session: aiohttp.ClientSession) -> Dict[str, Any]:
        """检查API服务是否可访问"""
        result = {"status": "unknown", "message": "", "details": {}}
        try:
            # 检查推理服务
            async with session.get(
                f"{SINGLEHOP_VLLM_BASE_URL.rstrip('/')}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    result["status"] = "ok"
                    result["message"] = "推理服务可访问"
                else:
                    result["status"] = "warning"
                    result["message"] = f"推理服务响应异常: {resp.status}"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"推理服务不可访问: {str(e)}"
            result["details"]["error"] = str(e)
        
        # 检查检索服务（如果启用）
        # 注意：检索服务可能没有/health端点，这是正常的，不影响主要功能
        if not DISABLE_RECALL:
            try:
                # 尝试调用实际的工具调用端点来检查服务是否可用
                async with session.post(
                    SINGLEHOP_RETRIEVAL_ENDPOINT,
                    json={"name": "retrieval_tool", "arguments": {"query": "test", "source_filter": "toollist", "top_k": 1}},
                    headers=SINGLEHOP_RETRIEVAL_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status in (200, 400, 422):  # 200成功，400/422是参数错误但服务可用
                        result["details"]["retrieval_service"] = "可访问"
                    else:
                        result["details"]["retrieval_service"] = f"响应异常: {resp.status}"
            except Exception as e:
                result["details"]["retrieval_service"] = f"不可访问（可能不影响评估）: {str(e)[:50]}"
        
        return result
    
    @staticmethod
    def check_data_format(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查数据格式是否符合要求"""
        result = {
            "status": "ok",
            "issues": [],
            "statistics": {"total": len(data), "valid": 0, "invalid": 0}
        }
        
        required_fields = ["conversations", "system"]
        for idx, conv in enumerate(data, 1):
            issues = []
            # 检查必需字段
            for field in required_fields:
                if field not in conv:
                    issues.append(f"缺少字段: {field}")
            
            # 检查conversations格式
            if "conversations" in conv:
                convs = conv["conversations"]
                if not isinstance(convs, list):
                    issues.append("conversations 不是列表")
                elif len(convs) < 2:
                    issues.append("conversations 长度不足（至少需要2条消息）")
                else:
                    # 检查是否有 human -> function_call 配对
                    has_pair = False
                    for i in range(len(convs) - 1):
                        if convs[i].get("from") == "human" and convs[i + 1].get("from") == "function_call":
                            has_pair = True
                            break
                    if not has_pair:
                        issues.append("未找到 human -> function_call 配对")
            
            if issues:
                result["status"] = "warning"
                result["issues"].append({"conversation_id": idx, "issues": issues})
                result["statistics"]["invalid"] += 1
            else:
                result["statistics"]["valid"] += 1
        
        return result
    
    @staticmethod
    def check_prompt_format(example: SingleHopExample) -> Dict[str, Any]:
        """检查Prompt格式是否正确"""
        result = {"status": "ok", "warnings": []}
        
        # 检查是否包含qwen3 template格式（可能不匹配）
        if "<|im_start|>" in example.system_prompt or "<|im_start|>" in example.user_prompt:
            result["status"] = "warning"
            result["warnings"].append("检测到qwen3 template格式，但singlehop脚本使用简单格式，可能不匹配")
        
        # 检查system prompt是否为空
        if not example.system_prompt or example.system_prompt.strip() == "":
            result["warnings"].append("system_prompt为空，将使用默认值")
        
        # 检查user prompt是否为空
        if not example.user_prompt or example.user_prompt.strip() == "":
            result["status"] = "error"
            result["warnings"].append("user_prompt为空，无法评估")
        
        return result
    
    @staticmethod
    def compare_with_multihop() -> Dict[str, Any]:
        """对比与multihop脚本的关键差异"""
        return {
            "prompt_format": {
                "singlehop": "简单格式：直接使用 system + user",
                "multihop": "qwen3 template格式：<|im_start|>system...<|im_end|>，需要解析提取",
                "difference": "singlehop假设数据是旧模板，multihop假设数据是qwen3 template格式"
            },
            "api_call": {
                "singlehop": "直接构建 messages: [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]",
                "multihop": "从qwen3格式中解析提取system和user，然后构建messages",
                "difference": "如果数据是qwen3格式，singlehop会直接传递template标签给API，可能导致格式错误"
            },
            "data_processing": {
                "singlehop": "_build_prompt: 简单拼接 system 和 user",
                "multihop": "parse_conversations: 构建完整的qwen3 template格式字符串",
                "difference": "singlehop不处理template标签，multihop会构建完整的template格式"
            },
            "concurrency": {
                "singlehop": "简单的semaphore控制（MAX_CONCURRENT_EXAMPLES）",
                "multihop": "多层并发控制（对话级、pair级、API级）",
                "difference": "multihop的并发控制更精细"
            }
        }


async def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger.add(args.log_file, rotation="50 MB")

    logger.info("=" * 60)
    logger.info("启动单跳评估脚本（旧模板）")
    logger.info("模型: {}", SINGLEHOP_MODEL_NAME)
    logger.info("推理服务: {}", SINGLEHOP_API_URL)
    logger.info("调用服务端口: {}", SERVICE_PORTS)
    logger.info("Recall评估: {}", "启用" if not DISABLE_RECALL else "禁用")
    logger.info("严格模式: {}", "启用" if args.strict_mode else "禁用（使用原有逻辑）")
    if args.strict_mode:
        logger.info("  ⚠️ 严格模式：参数评估仅在工具名匹配时进行，参数必须完全匹配才得分")
    logger.info("业务跳使用训练数据observation: {}", "是" if USE_TRAINING_DATA_OBSERVATION else "否（使用实际检索结果）")
    if USE_TRAINING_DATA_OBSERVATION:
        logger.info("  ⚠️  此模式用于测试过拟合效果，评估模型对训练数据的记忆程度")
        logger.info("  ⚠️  不适合评估模型的真实应用能力和泛化能力")
    logger.info("=" * 60)
    
    # ========== 自检机制 ==========
    logger.info("")
    logger.info("🔍 开始自检验证...")
    validator = SelfCheckValidator()
    
    # 1. 检查API服务
    logger.info("1️⃣ 检查API服务可访问性...")
    async with aiohttp.ClientSession() as session:
        api_check = await validator.check_api_service(session)
        if api_check["status"] == "ok":
            logger.info("   ✅ {}", api_check["message"])
        elif api_check["status"] == "warning":
            logger.warning("   ⚠️ {}", api_check["message"])
        else:
            logger.error("   ❌ {}", api_check["message"])
            logger.error("   💡 请检查推理服务是否运行在: {}", SINGLEHOP_VLLM_BASE_URL)
    
    # 2. 检查数据格式
    logger.info("2️⃣ 检查数据格式...")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        format_check = validator.check_data_format(data)
        logger.info("   📊 数据统计: 总计={}, 有效={}, 无效={}", 
                   format_check["statistics"]["total"],
                   format_check["statistics"]["valid"],
                   format_check["statistics"]["invalid"])
        if format_check["issues"]:
            logger.warning("   ⚠️ 发现 {} 个格式问题", len(format_check["issues"]))
            for issue in format_check["issues"][:5]:  # 只显示前5个
                logger.warning("      - 对话 {}: {}", issue["conversation_id"], ", ".join(issue["issues"]))
        else:
            logger.info("   ✅ 数据格式检查通过")
    except Exception as e:
        logger.error("   ❌ 数据文件读取失败: {}", str(e))
        return
    
    # 3. 检查Prompt格式（采样检查）
    logger.info("3️⃣ 检查Prompt格式（采样前3个样本）...")
    evaluator = SingleHopEvaluator(strict_mode=args.strict_mode)
    examples = evaluator.data_processor.parse(data[:3])  # 只检查前3个
    for ex in examples:
        prompt_check = validator.check_prompt_format(ex)
        if prompt_check["status"] == "error":
            logger.error("   ❌ Pair {}: {}", ex.pair_id, ", ".join(prompt_check["warnings"]))
        elif prompt_check["warnings"]:
            logger.warning("   ⚠️ Pair {}: {}", ex.pair_id, ", ".join(prompt_check["warnings"]))
        else:
            logger.info("   ✅ Pair {}: Prompt格式正确", ex.pair_id)
    
    # 4. 对比与multihop的差异
    logger.info("4️⃣ 对比与multihop脚本的关键差异...")
    comparison = validator.compare_with_multihop()
    logger.info("   📋 Prompt格式差异:")
    logger.info("      SingleHop: {}", comparison["prompt_format"]["singlehop"])
    logger.info("      MultiHop: {}", comparison["prompt_format"]["multihop"])
    logger.warning("      ⚠️ 差异: {}", comparison["prompt_format"]["difference"])
    logger.info("   📋 API调用差异:")
    logger.info("      SingleHop: {}", comparison["api_call"]["singlehop"])
    logger.info("      MultiHop: {}", comparison["api_call"]["multihop"])
    logger.warning("      ⚠️ 差异: {}", comparison["api_call"]["difference"])
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 自检完成，开始评估...")
    logger.info("=" * 60)
    logger.info("")
    
    # 执行评估
    diagnostic_mode = args.diagnostic
    eval_all_hops = args.eval_all_hops and not args.eval_retrieval_only
    
    if diagnostic_mode:
        logger.info("")
        logger.info("🔍 诊断模式已启用，将显示详细的样本和预测信息")
        logger.info("")
    
    logger.info("")
    logger.info("📊 评估模式: {}", "所有跳（检索跳+业务跳）" if eval_all_hops else "仅检索跳")
    logger.info("")
    
    results = await evaluator.evaluate_file(args.input_file, diagnostic_mode=diagnostic_mode, eval_all_hops=eval_all_hops)
    dump_report(results, args.output_file)

    summary = evaluator.summarize(results)
    logger.info("=" * 60)
    logger.info("📊 评估结果汇总")
    logger.info("=" * 60)
    logger.info("总样本数: {}", summary["total"])
    logger.info("总体 Accuracy: {:.3f}", summary["accuracy"])
    logger.info("总体 Precision@1: {:.3f}", summary["precision@1"])
    
    # 显示参数准确率（严格模式）
    if summary.get("arg_accuracy") is not None:
        logger.info("参数准确率 (严格模式): {:.3f} (可评估样本: {})", 
                   summary["arg_accuracy"], summary.get("arg_denominator", 0))
    elif args.strict_mode:
        logger.info("参数准确率 (严格模式): 暂无数据（无工具名匹配的样本）")
    
    # 分别显示检索跳和业务跳的统计
    if summary.get("retrieval_stats"):
        ret_stats = summary["retrieval_stats"]
        logger.info("")
        logger.info("🔍 检索工具调用统计:")
        logger.info("  样本数: {}", ret_stats["count"])
        logger.info("  Accuracy: {:.3f}", ret_stats["accuracy"])
        logger.info("  Precision@1: {:.3f}", ret_stats["precision@1"])
    
    if summary.get("business_stats"):
        bus_stats = summary["business_stats"]
        logger.info("")
        logger.info("💼 业务工具调用统计:")
        logger.info("  样本数: {}", bus_stats["count"])
        logger.info("  Accuracy: {:.3f}", bus_stats["accuracy"])
        logger.info("  Precision@1: {:.3f}", bus_stats["precision@1"])
    
    # 详细的Recall统计
    recall_samples = [r for r in results if r.recall is not None]
    recall_with_retrieval = [r for r in results if r.details.get("target_call", {}).get("name") == "retrieval_tool"]
    logger.info("")
    logger.info("📊 Recall统计详情:")
    logger.info("   - 总样本数: {}", summary["total"])
    logger.info("   - retrieval_tool调用数: {}", len(recall_with_retrieval))
    logger.info("   - 计算了Recall的样本数: {}", len(recall_samples))
    if recall_with_retrieval:
        logger.info("   - retrieval_tool调用中，满足Recall计算条件的: {}", len(recall_samples))
        if len(recall_samples) < len(recall_with_retrieval):
            logger.warning("   ⚠️ 有 {} 个retrieval_tool调用未计算Recall（可能因为工具名不匹配或缺少next_tool_name）", 
                          len(recall_with_retrieval) - len(recall_samples))
    
    if summary.get("recall@5") is not None:
        logger.info("   - Recall@5: {:.3f}", summary["recall@5"])
    else:
        logger.info("   - Recall@5: 暂无数据")
        if recall_with_retrieval:
            logger.warning("   ⚠️ 有retrieval_tool调用但未计算Recall，可能原因:")
            logger.warning("      1. 工具名预测不匹配（name_score != 1.0）")
            logger.warning("      2. 缺少next_tool_name（数据中没有后续业务工具）")
            logger.warning("      3. Recall计算被禁用")
    
    # 输出详细的失败原因统计
    if summary.get("failure_stats"):
        stats = summary["failure_stats"]
        logger.info("")
        logger.info("📊 失败原因统计:")
        logger.info("   ✅ 完全匹配: {}", stats["perfect_matches"])
        logger.info("   ❌ 工具名不匹配: {}", stats["tool_name_mismatches"])
        logger.info("   ⚠️ 参数不匹配: {}", stats["argument_mismatches"])
        logger.info("   🔴 解析失败:")
        logger.info("      - 目标解析失败: {}", stats["parse_failures"]["target_parse_failed"])
        logger.info("      - 预测解析失败: {}", stats["parse_failures"]["predict_parse_failed"])
        logger.info("      - 两者都失败: {}", stats["parse_failures"]["both_parse_failed"])
        
        # 如果工具名不匹配，显示前5个不匹配案例的详细信息
        if stats["tool_name_mismatches"] > 0:
            logger.warning("")
            logger.warning("⚠️ 发现工具名不匹配，检查前5个不匹配案例:")
            mismatch_count = 0
            for r in results:
                target_call = r.details.get("target_call", {})
                predict_call = r.details.get("predict_call", {})
                target_name = target_call.get("name", "未知")
                predict_name = predict_call.get("name", "未知")
                if target_name != predict_name and target_call and predict_call:
                    mismatch_count += 1
                    logger.warning("")
                    logger.warning("   案例 {} (pair_id: {}):", mismatch_count, r.pair_id)
                    logger.warning("      目标工具: {}", target_name)
                    logger.warning("      预测工具: {}", predict_name)
                    logger.warning("      目标内容: {}", r.target[:200])
                    logger.warning("      预测内容: {}", r.predict[:200])
                    if mismatch_count >= 5:
                        break
    
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


