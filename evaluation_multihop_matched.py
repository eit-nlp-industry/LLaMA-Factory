#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多跳模型评估脚本（与训练template完全匹配）
适配cutoff_len=10240和完整tools定义方案
"""

import json
import asyncio
import re
import sys
import os
import argparse
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import aiohttp

# 模型服务配置
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:5526")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "my_lora")
QWEN_API_URL = f"{VLLM_BASE_URL.rstrip('/')}/v1/chat/completions"

# 并发控制
MAX_CONCURRENT_CONVERSATIONS = int(os.getenv("MAX_CONCURRENT_CONVERSATIONS", "2"))
MAX_CONCURRENT_PAIRS = int(os.getenv("MAX_CONCURRENT_PAIRS", "5"))

@dataclass
class MultiHopPair:
    """多跳评估对"""
    pair_id: int
    hop_index: int
    hop_type: str  # 'retrieval' 或 'business_tool'
    source: str
    target: str
    conversation_id: int

@dataclass
class MultiHopResult:
    """多跳评估结果"""
    conversation_id: int
    pair_id: int
    hop_index: int
    hop_type: str
    target: str
    predict: str
    score: float
    tool_name_score: float

class MultiHopDataProcessor:
    """多跳数据处理 - 使用qwen3 template格式"""
    
    def parse_conversations(self, conversation_data: Dict, conversation_id: int) -> List[MultiHopPair]:
        """
        解析conversations，使用与训练时完全一致的qwen3 template格式
        """
        conversations = conversation_data["conversations"]
        system_prompt = conversation_data["system"]
        tools = conversation_data.get("tools", "[]")
        
        # 准备system内容（包含tools，用于Pair 1）
        try:
            tools_str = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
        except:
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
        
        # 提取原始用户query
        original_query = ""
        for msg in conversations:
            if msg["from"] == "human":
                original_query = msg["value"]
                break
        
        pairs = []
        pair_id = 1
        hop_index = 1
        
        i = 0
        while i < len(conversations):
            msg = conversations[i]
            
            if msg["from"] == "human":
                # Pair 1: human -> function_call
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "function_call":
                    # 使用qwen3 template格式
                    source = f"<|im_start|>system\n{base_system_with_tools}<|im_end|>\n<|im_start|>user\n{msg['value']}<|im_end|>\n<|im_start|>assistant\n"
                    target = conversations[i + 1]["value"]
                    
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
                        conversation_id=conversation_id
                    ))
                    pair_id += 1
                    hop_index += 1
                    i += 2
                else:
                    i += 1
            
            elif msg["from"] == "observation":
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
                        # 使用format_tools处理（模拟template的行为）
                        # 实际上tools_str就是格式化后的结果
                        observation_content = f"[可用工具定义]\n{tools_str}\n\n{observation_content}"
                    
                    # 使用qwen3的format_observation格式：user角色 + <tool_response>标签
                    source = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<tool_response>\n{observation_content}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
                    target = conversations[i + 1]["value"]
                    
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
                        conversation_id=conversation_id
                    ))
                    pair_id += 1
                    hop_index += 1
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return pairs

class LLMPredictor:
    """LLM预测模块"""
    
    def __init__(self, model_type: str = "qwen3"):
        self.model_type = QWEN_MODEL_NAME
        self.max_retries = 5
    
    async def call_qwen_api(self, session: aiohttp.ClientSession, prompt: List[Dict]) -> str:
        """异步调用Qwen API"""
        headers = {"Content-Type": "application/json"}
        if VLLM_API_KEY:
            headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
        
        data = {
            "model": self.model_type,
            "messages": prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}
        }
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(QWEN_API_URL, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE)
                        return content.strip()
                    else:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            raise Exception(f"API调用失败: {response.status}")
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        return ""
    
    async def predict(self, session: aiohttp.ClientSession, source: str) -> str:
        """预测 - 从qwen3格式的source中提取system和user内容"""
        try:
            # 提取system部分
            system_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', source, re.DOTALL)
            system_content = system_match.group(1) if system_match else ""
            
            # 提取user部分（qwen3的observation也用user角色）
            user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', source, re.DOTALL)
            user_content = user_match.group(1) if user_match else ""
            
            prompt = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            return await self.call_qwen_api(session, prompt)
        except Exception as e:
            print(f"预测失败: {e}")
            return ""

class ToolCallEvaluator:
    """工具调用评估"""
    
    def extract_tool_call(self, text: str) -> Dict[str, Any]:
        """提取工具调用"""
        try:
            if text.startswith('{') and text.endswith('}'):
                return json.loads(text)
            
            match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            return json.loads(text)
        except:
            return {}
    
    def evaluate_tool_call(self, target: str, predict: str) -> Tuple[float, float, Dict]:
        """评估工具调用"""
        target_call = self.extract_tool_call(target)
        predict_call = self.extract_tool_call(predict)
        
        details = {
            "target_call": target_call,
            "predict_call": predict_call,
            "tool_name_match": False,
            "arguments_match": False
        }
        
        score = 0.0
        tool_name_score = 0.0
        
        if "name" in target_call:
            target_name = target_call.get("name", "")
            predict_name = predict_call.get("name", "")
            if target_name == predict_name and target_name:
                details["tool_name_match"] = True
                score += 0.5
                tool_name_score = 1.0
            
            target_args = target_call.get("arguments", {}) or {}
            predict_args = predict_call.get("arguments", {}) or {}
            if target_args and predict_args:
                matching_args = sum(1 for k, v in target_args.items() if predict_args.get(k) == v)
                arg_score = matching_args / len(target_args) if target_args else 0
                details["arguments_match"] = (arg_score == 1.0)
                score += 0.5 * arg_score
        
        return score, tool_name_score, details

class MultiHopEvaluator:
    """多跳评估主类"""
    
    def __init__(self):
        self.data_processor = MultiHopDataProcessor()
        self.llm_predictor = LLMPredictor()
        self.tool_evaluator = ToolCallEvaluator()
    
    async def evaluate_single_pair(self, session: aiohttp.ClientSession, pair: MultiHopPair) -> MultiHopResult:
        """评估单个pair"""
        predict = await self.llm_predictor.predict(session, pair.source)
        score, tool_name_score, details = self.tool_evaluator.evaluate_tool_call(pair.target, predict)
        
        return MultiHopResult(
            conversation_id=pair.conversation_id,
            pair_id=pair.pair_id,
            hop_index=pair.hop_index,
            hop_type=pair.hop_type,
            target=pair.target,
            predict=predict,
            score=score,
            tool_name_score=tool_name_score
        )
    
    async def evaluate_file(self, file_path: str, start_idx: int = 0, end_idx: Optional[int] = None) -> List[MultiHopResult]:
        """评估整个文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if end_idx is None:
            end_idx = len(data)
        
        all_results = []
        
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            for idx, conversation_data in enumerate(data[start_idx:end_idx], start=start_idx):
                print(f"评估对话 {idx + 1}/{end_idx}")
                pairs = self.data_processor.parse_conversations(conversation_data, idx + 1)
                
                for pair in pairs:
                    result = await self.evaluate_single_pair(session, pair)
                    all_results.append(result)
                    print(f"  Hop {pair.hop_index} ({pair.hop_type}): acc={result.score:.3f}, prec@1={result.tool_name_score:.3f}")
        
        return all_results
    
    def calculate_metrics(self, results: List[MultiHopResult]) -> Dict:
        """计算指标"""
        by_hop = defaultdict(lambda: {"total": 0, "scores": [], "tool_name_scores": []})
        
        for r in results:
            by_hop[r.hop_index]["total"] += 1
            by_hop[r.hop_index]["scores"].append(r.score)
            by_hop[r.hop_index]["tool_name_scores"].append(r.tool_name_score)
        
        metrics = {}
        for hop_idx, data in by_hop.items():
            metrics[str(hop_idx)] = {
                "total": data["total"],
                "accuracy": sum(data["scores"]) / data["total"] if data["total"] > 0 else 0.0,
                "precision@1": sum(data["tool_name_scores"]) / data["total"] if data["total"] > 0 else 0.0
            }
        
        return {
            "by_hop_metrics": metrics,
            "overall": {
                "total": len(results),
                "accuracy": sum(r.score for r in results) / len(results) if results else 0.0,
                "precision@1": sum(r.tool_name_score for r in results) / len(results) if results else 0.0
            }
        }

def parse_args():
    parser = argparse.ArgumentParser(description="多跳模型评估脚本")
    parser.add_argument("--input_file", "-i", type=str, 
                       default="data/dataset/10_30/multihop_retrieval_test.json",
                       help="输入JSON文件路径")
    parser.add_argument("--output_file", "-o", type=str,
                       default="evaluation_multihop_results.json",
                       help="输出结果文件路径")
    parser.add_argument("--start_idx", "-s", type=int, default=0)
    parser.add_argument("--end_idx", "-e", type=int, default=None)
    return parser.parse_args()

async def main():
    args = parse_args()
    
    print("="*80)
    print("多跳模型评估")
    print("="*80)
    print(f"模型: {QWEN_MODEL_NAME}")
    print(f"输入: {args.input_file}")
    print(f"输出: {args.output_file}")
    print(f"格式: qwen3 template + 完整tools定义")
    print("="*80)
    
    evaluator = MultiHopEvaluator()
    results = await evaluator.evaluate_file(args.input_file, args.start_idx, args.end_idx)
    
    metrics = evaluator.calculate_metrics(results)
    
    report = {
        "summary": metrics,
        "model": QWEN_MODEL_NAME,
        "cutoff_len": 10240,
        "template": "qwen3_with_full_tools"
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print()
    print("="*80)
    print("评估完成！")
    print("="*80)
    print(f"总评估对: {metrics['overall']['total']}")
    print(f"总体准确率: {metrics['overall']['accuracy']:.3f}")
    print(f"总体precision@1: {metrics['overall']['precision@1']:.3f}")
    print()
    print("按Hop统计:")
    for hop, m in sorted(metrics['by_hop_metrics'].items(), key=lambda x: int(x[0])):
        print(f"  Hop {hop}: acc={m['accuracy']:.3f}, prec@1={m['precision@1']:.3f}, n={m['total']}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())

