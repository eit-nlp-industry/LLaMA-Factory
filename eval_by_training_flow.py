#!/usr/bin/env python3
"""
基于真实训练流程的测试集评测脚本
按照训练时的pair分割方式进行评估，包括function call评估和LLM judge评估
"""

import json
import os
import requests
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer
from src.llamafactory.data.template import TEMPLATES

class TrainingFlowEvaluator:
    def __init__(self, model_endpoint="http://localhost:8021/v1/completions", 
                 judge_endpoint="http://localhost:8021/v1/completions"):
        """
        初始化评估器
        
        Args:
            model_endpoint: 被评估模型的API端点
            judge_endpoint: LLM judge模型的API端点
        """
        self.model_endpoint = model_endpoint
        self.judge_endpoint = judge_endpoint
        
        # 加载tokenizer和template
        model_name = "/data/models/Qwen3-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.template = TEMPLATES["qwen3"]
        self.template.fix_special_tokens(self.tokenizer)
        
        self.log_file = "/home/ziqiang/LLaMA-Factory/eval_results.log"
        self._init_log()
    
    def _init_log(self):
        """初始化日志文件"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.log_debug("=== 评测开始 ===")
    
    def log_debug(self, msg):
        """记录调试信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} | INFO | {msg}\n"
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
            f.flush()
        
        print(f"EVAL | {msg}")
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """加载测试数据"""
        with open(test_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.log_debug(f"加载测试数据: {len(data)} 条样本")
        return data
    
    def parse_conversations_to_messages(self, conversations: List[Dict]) -> List[Dict]:
        """将conversations转换为messages格式"""
        messages = []
        for conv in conversations:
            if conv["from"] == "human":
                messages.append({"role": "user", "content": conv["value"]})
            elif conv["from"] == "gpt":
                messages.append({"role": "assistant", "content": conv["value"]})
            elif conv["from"] == "function_call":
                messages.append({"role": "function", "content": conv["value"]})
            elif conv["from"] == "observation":
                messages.append({"role": "observation", "content": conv["value"]})
        return messages
    
    def split_into_training_pairs(self, messages: List[Dict], system: str = "", tools: str = "") -> List[Tuple[str, str, str]]:
        """
        按照训练流程将对话分割成pairs
        修改后：每个pair的source都包含原始用户的query
        
        Returns:
            List[Tuple[source_text, target_text, pair_type]]
            pair_type: 'function_call' 或 'assistant_response'
        """
        pairs = []
        
        # 提取原始用户query
        original_user_message = None
        for msg in messages:
            if msg["role"] == "user":
                original_user_message = msg
                break
        
        if not original_user_message:
            return pairs
        
        # 遍历messages，找到需要评估的pairs
        i = 0
        while i < len(messages):
            # 寻找用户消息作为source
            if messages[i]["role"] == "user":
                user_message = messages[i]
                
                # 检查下一个消息是否为function call
                if i + 1 < len(messages) and messages[i + 1]["role"] == "function":
                    # 第一个pair: system + tools + user -> function_call
                    function_message = messages[i + 1]
                    pair_type = "function_call"
                    i += 2  # 跳过user和function消息
                    
                    # 构建source: system + tools + user
                    source_messages = [user_message]
                    
                    # 使用template编码，将system和tools作为参数传递
                    encoded_source = self.template._encode(self.tokenizer, source_messages, system, tools)
                    encoded_target = self.template._encode(self.tokenizer, [function_message], "", "")
                    
                    # 解码为文本 - template._encode返回的是token ID列表
                    if isinstance(encoded_source, list) and len(encoded_source) > 0:
                        source_text = self.tokenizer.decode(encoded_source[0], skip_special_tokens=False)
                    else:
                        source_text = ""
                    
                    if isinstance(encoded_target, list) and len(encoded_target) > 0:
                        target_text = self.tokenizer.decode(encoded_target[0], skip_special_tokens=False)
                    else:
                        target_text = ""
                    
                    pairs.append((source_text, target_text, pair_type))
                    
                    # 检查是否有observation和assistant消息
                    if i < len(messages) and messages[i]["role"] == "observation":
                        observation_message = messages[i]
                        i += 1  # 跳过observation消息
                        
                        if i < len(messages) and messages[i]["role"] == "assistant":
                            # 第二个pair: system + user + observation -> assistant_response
                            assistant_message = messages[i]
                            pair_type = "assistant_response"
                            i += 1  # 跳过assistant消息
                            
                            # 构建source: system + user + observation (包含原始query)
                            source_messages = [original_user_message, observation_message]
                            
                            # 使用template编码，将system作为参数传递（不包含tools）
                            encoded_source = self.template._encode(self.tokenizer, source_messages, system, "")
                            encoded_target = self.template._encode(self.tokenizer, [assistant_message], "", "")
                            
                            # 解码为文本 - template._encode返回的是token ID列表
                            if isinstance(encoded_source, list) and len(encoded_source) > 0:
                                source_text = self.tokenizer.decode(encoded_source[0], skip_special_tokens=False)
                            else:
                                source_text = ""
                            
                            if isinstance(encoded_target, list) and len(encoded_target) > 0:
                                target_text = self.tokenizer.decode(encoded_target[0], skip_special_tokens=False)
                            else:
                                target_text = ""
                            
                            pairs.append((source_text, target_text, pair_type))
                
                elif i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                    # 直接user -> assistant的情况（没有function call）
                    assistant_message = messages[i + 1]
                    pair_type = "assistant_response"
                    i += 2  # 跳过user和assistant消息
                    
                    # 构建source: system + user
                    source_messages = [user_message]
                    
                    # 使用template编码
                    encoded_source = self.template._encode(self.tokenizer, source_messages, system, tools)
                    encoded_target = self.template._encode(self.tokenizer, [assistant_message], "", "")
                    
                    # 解码为文本 - template._encode返回的是token ID列表
                    if isinstance(encoded_source, list) and len(encoded_source) > 0:
                        source_text = self.tokenizer.decode(encoded_source[0], skip_special_tokens=False)
                    else:
                        source_text = ""
                    
                    if isinstance(encoded_target, list) and len(encoded_target) > 0:
                        target_text = self.tokenizer.decode(encoded_target[0], skip_special_tokens=False)
                    else:
                        target_text = ""
                    
                    pairs.append((source_text, target_text, pair_type))
                else:
                    # 没有找到对应的target消息，跳过
                    i += 1
                    continue
            else:
                i += 1
        
        return pairs
    
    def call_model_api(self, prompt: str, max_tokens: int = 2048) -> str:
        """调用模型API - 参考llm_calling.py的chat_template_kwargs参数"""
        try:
            # 使用prompt格式，确保禁用thinking模式
            payload = {
                "model": "my_lora",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "enable_thinking": False,  # 明确禁用thinking模式
                #"stop": ["<think>", "</think>"]  # 添加停止词以防止thinking输出
            }
            
            response = requests.post(self.model_endpoint, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # 使用text格式，因为这个API返回的是text
                response_text = result["choices"][0]["text"]
                
                # 详细记录API返回内容
                self.log_debug(f"API调用成功 - 原始响应长度: {len(response_text)}")
                self.log_debug(f"API调用成功 - 原始响应内容: '{response_text}'")
                
                # 检查是否包含thinking内容并强制移除
                original_length = len(response_text)
                if "<think>" in response_text or "</think>" in response_text:
                    self.log_debug("⚠️ 检测到thinking内容，强制移除")
                    # 移除thinking标签及其内容
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                    # 移除残留的thinking标签
                    response_text = re.sub(r'<think>.*$', '', response_text, flags=re.DOTALL)
                    response_text = re.sub(r'^.*</think>', '', response_text, flags=re.DOTALL)
                    response_text = response_text.replace('<think>', '').replace('</think>', '')
                    self.log_debug(f"🧹 已移除thinking内容 - 原始长度: {original_length}, 处理后长度: {len(response_text)}")
                else:
                    self.log_debug("✅ 未检测到thinking内容")
                
                # 检查是否为空
                if not response_text.strip():
                    self.log_debug("⚠️ 警告: API返回了空内容")
                
                return response_text
            else:
                self.log_debug(f"API调用失败: {response.status_code}")
                return ""
                
        except Exception as e:
            self.log_debug(f"API调用异常: {str(e)}")
            return ""
    
    def extract_function_call(self, text: str) -> Dict:
        """从文本中提取function call信息"""
        # 移除思维链部分
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 处理转义字符
        text = text.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        
        # 提取tool_call内容
        tool_call_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
        if tool_call_match:
            try:
                json_content = tool_call_match.group(1).strip()
                func_json = json.loads(json_content)
                return {
                    "name": func_json.get("name", ""),
                    "arguments": func_json.get("arguments", {})
                }
            except json.JSONDecodeError as e:
                self.log_debug(f"JSON解析失败: {str(e)}, 内容: {json_content[:100]}...")
        
        # 尝试直接查找JSON格式的function call
        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
        json_match = re.search(json_pattern, text, re.DOTALL)
        if json_match:
            try:
                json_content = json_match.group(0)
                # 处理可能的多层转义
                json_content = json_content.replace('\\n', '\n').replace('\\"', '"')
                func_json = json.loads(json_content)
                return {
                    "name": func_json.get("name", ""),
                    "arguments": func_json.get("arguments", {})
                }
            except json.JSONDecodeError:
                pass
        
        return {"name": "", "arguments": {}}
    
    def evaluate_function_call(self, predicted: str, expected: str) -> Dict[str, Any]:
        """评估function call的准确性"""
        pred_func = self.extract_function_call(predicted)
        exp_func = self.extract_function_call(expected)
        
        # 工具名称匹配
        name_match = pred_func["name"] == exp_func["name"]
        
        # 参数匹配 - 简化版本，检查关键参数
        args_match = True
        missing_args = []
        wrong_args = []
        
        for key, value in exp_func["arguments"].items():
            if key not in pred_func["arguments"]:
                missing_args.append(key)
                args_match = False
            elif pred_func["arguments"][key] != value:
                wrong_args.append(key)
                args_match = False
        
        return {
            "name_match": name_match,
            "args_match": args_match,
            "overall_match": name_match and args_match,
            "predicted_name": pred_func["name"],
            "expected_name": exp_func["name"],
            "missing_args": missing_args,
            "wrong_args": wrong_args
        }
    
    def llm_judge_response(self, predicted: str, expected: str, context: str = "") -> Dict[str, Any]:
        """使用LLM judge评估助手回复"""
        judge_prompt = f"""请作为一个公正的裁判，评估AI助手的回复质量。

上下文信息：
{context}

期望回复：
{expected}

实际回复：
{predicted}

请从以下维度评分（1-5分）：
1. 准确性：回复内容是否准确无误
2. 完整性：是否涵盖了期望回复的主要信息
3. 相关性：回复是否切题，与问题相关
4. 清晰度：表述是否清晰易懂

请按以下格式输出：
准确性分数：X/5
完整性分数：X/5  
相关性分数：X/5
清晰度分数：X/5
总体分数：X/5
评估理由：[简要说明评分理由]"""

        try:
            # 使用prompt格式，确保禁用thinking模式
            payload = {
                "model": "my_lora",
                "prompt": judge_prompt,
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0,
                "enable_thinking": False,  # 明确禁用thinking模式
                #"stop": ["<think>", "</think>"]  # 添加停止词以防止thinking输出
            }
            self.log_debug(f"Payload sent: {json.dumps(payload)}")
            response = requests.post(self.judge_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()["choices"][0]["text"]
                
                # 移除thinking内容
                if "<think>" in result or "</think>" in result:
                    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                    result = re.sub(r'<think>.*$', '', result, flags=re.DOTALL)
                    result = re.sub(r'^.*</think>', '', result, flags=re.DOTALL)
                    result = result.replace('<think>', '').replace('</think>', '')
                
                # 解析评分结果
                scores = {}
                lines = result.split('\n')
                for line in lines:
                    if '分数：' in line:
                        if '准确性' in line:
                            scores['accuracy'] = self._extract_score(line)
                        elif '完整性' in line:
                            scores['completeness'] = self._extract_score(line)
                        elif '相关性' in line:
                            scores['relevance'] = self._extract_score(line)
                        elif '清晰度' in line:
                            scores['clarity'] = self._extract_score(line)
                        elif '总体' in line:
                            scores['overall'] = self._extract_score(line)
                
                return scores
            else:
                self.log_debug(f"LLM Judge API调用失败: {response.status_code}")
                return {}
        except Exception as e:
            self.log_debug(f"LLM Judge异常: {str(e)}")
            return {}
    
    def _extract_score(self, text: str) -> float:
        """从文本中提取分数"""
        match = re.search(r'(\d+(?:\.\d+)?)/5', text)
        if match:
            return float(match.group(1))
        return 0.0
    
    def evaluate_sample(self, sample: Dict) -> Dict[str, Any]:
        """评估单个样本"""
        conversations = sample["conversations"]
        system = sample.get("system", "")
        tools = sample.get("tools", "")
        
        # 转换为messages格式
        messages = self.parse_conversations_to_messages(conversations)
        
        # 分割成训练pairs
        pairs = self.split_into_training_pairs(messages, system, tools)
        
        results = {
            "total_pairs": len(pairs),
            "function_call_results": [],
            "assistant_response_results": [],
            "overall_stats": {}
        }
        
        self.log_debug(f"样本包含 {len(pairs)} 个训练pairs")
        
        for i, (source_text, expected_target, pair_type) in enumerate(pairs):
            self.log_debug(f"评估 Pair {i+1}/{len(pairs)} (类型: {pair_type})")
            
            # 调用模型生成
            predicted_target = self.call_model_api(source_text)
            
            if pair_type == "function_call":
                # 评估function call
                eval_result = self.evaluate_function_call(predicted_target, expected_target)
                eval_result["pair_index"] = i
                eval_result["source_preview"] = source_text[:1000]  # 增加预览长度
                eval_result["source_length"] = len(source_text)
                eval_result["predicted_full"] = predicted_target
                eval_result["expected_full"] = expected_target
                results["function_call_results"].append(eval_result)
                
                self.log_debug(f"  Function Call - 名称匹配: {eval_result['name_match']}, 参数匹配: {eval_result['args_match']}")
                
            elif pair_type == "assistant_response":
                # 评估助手回复
                judge_result = self.llm_judge_response(predicted_target, expected_target, source_text[:200])
                eval_result = {
                    "pair_index": i,
                    "judge_scores": judge_result,
                    "source_preview": source_text[:1000],  # 增加预览长度
                    "source_length": len(source_text),
                    "predicted_full": predicted_target,
                    "expected_full": expected_target
                }
                results["assistant_response_results"].append(eval_result)
                
                overall_score = judge_result.get('overall', 0)
                self.log_debug(f"  Assistant Response - 总体分数: {overall_score}/5")
        
        return results
    
    def evaluate_dataset(self, test_file: str, output_file: str = None) -> Dict[str, Any]:
        """评估整个数据集"""
        # 加载测试数据
        test_data = self.load_test_data(test_file)
        
        all_results = []
        function_call_stats = {"total": 0, "correct_name": 0, "correct_args": 0, "fully_correct": 0}
        assistant_response_stats = {"total": 0, "scores": []}
        
        for i, sample in enumerate(test_data):
            self.log_debug(f"\n=== 评估样本 {i+1}/{len(test_data)} ===")
            
            try:
                result = self.evaluate_sample(sample)
                all_results.append(result)
                
                # 统计function call结果
                for fc_result in result["function_call_results"]:
                    function_call_stats["total"] += 1
                    if fc_result["name_match"]:
                        function_call_stats["correct_name"] += 1
                    if fc_result["args_match"]:
                        function_call_stats["correct_args"] += 1
                    if fc_result["overall_match"]:
                        function_call_stats["fully_correct"] += 1
                
                # 统计assistant response结果
                for ar_result in result["assistant_response_results"]:
                    assistant_response_stats["total"] += 1
                    overall_score = ar_result["judge_scores"].get("overall", 0)
                    assistant_response_stats["scores"].append(overall_score)
                
            except Exception as e:
                self.log_debug(f"样本 {i+1} 评估失败: {str(e)}")
                continue
            
            # 每10个样本保存一次中间结果
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(all_results, function_call_stats, assistant_response_stats, i + 1)
        
        # 计算最终统计
        final_stats = self._calculate_final_stats(function_call_stats, assistant_response_stats)
        
        # 保存结果
        if output_file:
            self._save_final_results(all_results, final_stats, output_file)
        
        self.log_debug("=== 评测完成 ===")
        return {"results": all_results, "stats": final_stats}
    
    def _calculate_final_stats(self, fc_stats: Dict, ar_stats: Dict) -> Dict:
        """计算最终统计信息"""
        final_stats = {
            "function_call": {
                "total": fc_stats["total"],
                "name_accuracy": fc_stats["correct_name"] / max(fc_stats["total"], 1),
                "args_accuracy": fc_stats["correct_args"] / max(fc_stats["total"], 1),
                "overall_accuracy": fc_stats["fully_correct"] / max(fc_stats["total"], 1)
            },
            "assistant_response": {
                "total": ar_stats["total"],
                "average_score": sum(ar_stats["scores"]) / max(len(ar_stats["scores"]), 1) if ar_stats["scores"] else 0,
                "score_distribution": self._get_score_distribution(ar_stats["scores"])
            }
        }
        
        self.log_debug(f"Function Call 总体准确率: {final_stats['function_call']['overall_accuracy']:.2%}")
        self.log_debug(f"Assistant Response 平均分数: {final_stats['assistant_response']['average_score']:.2f}/5")
        
        return final_stats
    
    def _get_score_distribution(self, scores: List[float]) -> Dict:
        """获取分数分布"""
        if not scores:
            return {}
        
        bins = {"1.0-2.0": 0, "2.0-3.0": 0, "3.0-4.0": 0, "4.0-5.0": 0}
        for score in scores:
            if score < 2.0:
                bins["1.0-2.0"] += 1
            elif score < 3.0:
                bins["2.0-3.0"] += 1
            elif score < 4.0:
                bins["3.0-4.0"] += 1
            else:
                bins["4.0-5.0"] += 1
        
        return bins
    
    def _save_intermediate_results(self, results: List, fc_stats: Dict, ar_stats: Dict, completed: int):
        """保存中间结果"""
        self.log_debug(f"已完成 {completed} 个样本的评估")
        # 可以在这里保存中间结果到文件
    
    def _save_final_results(self, results: List, stats: Dict, output_file: str):
        """保存最终结果"""
        output_data = {
            "evaluation_results": results,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.log_debug(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    # 配置
    test_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo.json"  # 真实测试数据文件路径
    output_file = "/home/ziqiang/LLaMA-Factory/evaluation_results.json"
    
    # 创建评估器
    evaluator = TrainingFlowEvaluator()
    
    # 执行评估
    results = evaluator.evaluate_dataset(test_file, output_file)
    
    print("评估完成！")
    print(f"详细日志: {evaluator.log_file}")
    print(f"结果文件: {output_file}")

if __name__ == "__main__":
    main()
