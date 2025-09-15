#!/usr/bin/env python3
"""
Function Call数据格式转换脚本
将分散在多个JSON文件中的数据合并转换为LLaMA-Factory支持的Function Call训练格式
"""

import json
import re
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def load_json_file(file_path: str) -> List[Dict]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def extract_function_call_from_output(output: str) -> Optional[str]:
    """从retrieval_function_call.json的output字段中提取function call JSON"""
    # 去除三引号包装
    output = output.strip('"""').strip()
    
    # 提取tool_call标签内的内容
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    match = re.search(pattern, output, re.DOTALL)
    
    if match:
        return match.group(1)
    else:
        print(f"Warning: Could not extract function call from output: {output[:100]}...")
        return None


def extract_tools_from_context(context_str: str) -> List[Dict]:
    """从context字符串中提取工具定义"""
    try:
        # 解析context字符串
        context_dict = eval(context_str)  # 注意：这里使用eval是不安全的，但context看起来是Python字典字符串
        tool_lists = context_dict.get('tool_lists', [])
        return tool_lists
    except Exception as e:
        print(f"Error parsing context: {e}")
        return []


def build_system_message() -> str:
    """构建系统消息"""
    return """

# 工具

你可以调用一个或多个函数来协助处理用户查询。

在 <tools></tools> XML 标签中提供了可用的函数签名：
<tools>
</tools>

你在输出时必须严格遵循以下规则：

1. 如果需要调用函数，则 **只能输出一个函数调用**，格式如下：
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

2. 如果你已经从工具返回结果或已有推理得出足够信息，必须立即停止调用工具，并输出最终答案，格式如下：
<answer>
你的最终答案在这里
</answer>

3. **智能流程阶段判断**：
- 仔细分析下方的对话流程历史，了解当前处于哪个阶段
- 如果看到Assistant已经调用过工具且User已经提供了<tool_response>...</tool_response>，说明工具调用已完成
- 如果工具返回空数据（如总收入为0、空列表等），应生成解释性答案而不是重复调用
- 如果已经获得足够信息回答用户问题，立即生成最终答案

4. **严格禁止以下行为**：
- 在同一轮输出中同时给出函数调用和最终答案  
- 使用完全相同的参数重复调用同一个工具
- 在工具已经返回结果（包括空结果）后，继续调用相同工具
- 忽略对话流程历史中已有的工具调用和响应信息

记住：基于对话流程历史判断当前阶段，一旦能够生成答案就立即输出，避免无意义的工具重复调用。"""


def find_matching_data(query: str, data_list: List[Dict], query_field: str = "query") -> Optional[Dict]:
    """根据query找到匹配的数据项"""
    for item in data_list:
        if item.get(query_field) == query:
            return item
    return None


def convert_tool_call_response_to_function_call(tool_data: Dict) -> Optional[str]:
    """将tool_call_tool_response.json中的数据转换为function_call格式"""
    try:
        params = tool_data.get("data", {}).get("params", {})
        if params:
            function_call = {
                "name": params.get("name"),
                "arguments": params.get("arguments", {})
            }
            return json.dumps(function_call, ensure_ascii=False)
    except Exception as e:
        print(f"Error converting tool call response: {e}")
    return None


def extract_tool_response_result(tool_resp: Dict) -> str:
    """安全地提取工具响应结果，只保留最内层的result内容"""
    try:
        response = tool_resp.get("response", {})
        
        # 检查response是否是字典
        if isinstance(response, dict):
            result = response.get("result", {})
            
            # 检查result是否是字典且包含result字段（去掉外层的format、jsonrpc、id等）
            if isinstance(result, dict) and "result" in result:
                # 只返回最内层的result内容
                inner_result = result["result"]
                return json.dumps(inner_result, ensure_ascii=False)
            else:
                # 如果result本身就是最终结果
                return json.dumps(result, ensure_ascii=False)
        else:
            # 如果response不是字典，直接返回
            return json.dumps(response, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error extracting tool response result: {e}")
        return json.dumps({"error": "Failed to extract result"}, ensure_ascii=False)


def split_train_test(data: List[Dict], train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """将数据按比例分割为训练集和测试集"""
    random.seed(random_seed)
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    train_size = int(len(data_copy) * train_ratio)
    train_data = data_copy[:train_size]
    test_data = data_copy[train_size:]
    
    return train_data, test_data


def convert_to_function_call_format(
    retrieval_fc_path: str,
    retrieval_response_path: str, 
    tool_response_path: str,
    answer_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
):
    """主转换函数"""
    
    # 加载所有数据文件
    print("Loading data files...")
    retrieval_fc_data = load_json_file(retrieval_fc_path)
    retrieval_response_data = load_json_file(retrieval_response_path)
    tool_response_data = load_json_file(tool_response_path)
    answer_data = load_json_file(answer_path)
    
    print(f"Loaded {len(retrieval_fc_data)} retrieval function calls")
    print(f"Loaded {len(retrieval_response_data)} retrieval responses")
    print(f"Loaded {len(tool_response_data)} tool responses")
    print(f"Loaded {len(answer_data)} answers")
    
    # 构建工具定义
    tools_definition = []
    if retrieval_fc_data:
        # 从第一条数据中提取工具定义
        tools_from_context = extract_tools_from_context(retrieval_fc_data[0].get("context", ""))
        tools_definition.extend(tools_from_context)
    
    # 从tool_response_data中添加更多工具定义（如analyze_revenue_by_type等）
    # 这里需要根据实际情况手动添加工具定义
    additional_tools = [
  {
        "name": "retrieval_tool",
        "chinese_name": "retrieval_tool",
        "description": "根据用户的问题，在知识库中搜索相关信息。可以指定知识来源（如工具库、对话历史或具体的'建德'、'新昌'文档库），并返回最匹配的结果。",
        "category": "nlp",
        "enabled": True,
        "source": "internal_service",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "用户的查询内容或问题",
              "label": "用户的查询内容或问题",
              "format": None,
              "pattern": None,
              "examples": None,
              "default": None,
              "enum": None
            },
            "top_k": {
              "type": "integer",
              "description": "可选：需要返回的最相关结果的数量",
              "label": "可选：需要返回的最相关结果的数量",
              "format": None,
              "pattern": None,
              "examples": None,
              "default": 3,
              "enum": None
            },
            "source_filter": {
              "type": "string",
              "description": "必选：指定检索的知识库来源以缩小搜索范围。'toollist'搜索mcp工具库，'xinchang'搜索新昌的导游手册，'jiande'搜索建德的导游手册。",
              "label": "必选：指定检索的知识库来源以缩小搜索范围。'toollist'搜索mcp工具库，'xinchang'搜索新昌的导游手册，'jiande'搜索建德的导游手册。",
              "format": None,
              "pattern": None,
              "examples": [
                "toollist",
                "jiande"
              ],
              "default": None,
              "enum": [
                "toollist",
                "jiande",
                "xinchang"
              ]
            },
            "user_id": {
              "type": "integer",
              "description": "必选，用户的ID，用于确认身份",
              "label": "必选，用户的ID，用于确认身份",
              "format": None,
              "pattern": None,
              "examples": None,
              "default": None,
              "enum": None
            }
          },
          "required": [
            "query",
            "source_filter",
            "user_id"
          ],
          "additionalProperties": None
        }
      }
    ]
    tools_definition.extend(additional_tools)
    
    # 转换数据
    converted_data = []
    
    for fc_item in retrieval_fc_data:
        query = fc_item.get("input", "")
        if not query:
            continue
            
        print(f"Processing query: {query[:50]}...")
        
        # 1. 提取第一个function call
        first_fc = extract_function_call_from_output(fc_item.get("output", ""))
        if not first_fc:
            print(f"Skipping query due to missing function call: {query[:50]}...")
            continue
        
        # 2. 找到对应的retrieval response
        retrieval_resp = find_matching_data(query, retrieval_response_data)
        if not retrieval_resp:
            print(f"Warning: No retrieval response found for query: {query[:50]}...")
            continue
        
        # 3. 找到对应的tool response
        tool_resp = find_matching_data(query, tool_response_data)
        if not tool_resp:
            print(f"Warning: No tool response found for query: {query[:50]}...")
            continue
        
        # 4. 找到对应的answer (这里需要根据实际的input字段匹配)
        answer_item = None
        for ans in answer_data:
            # 尝试匹配answer中的input字段与query
            if query in ans.get("input", "") or ans.get("input", "") in query:
                answer_item = ans
                break
        
        if not answer_item:
            print(f"Warning: No answer found for query: {query[:50]}...")
            # 使用默认回答
            final_answer = "抱歉，暂时无法提供详细分析。"
        else:
            final_answer = answer_item.get("output", "")
        
        # 5. 生成第二个function call
        second_fc = convert_tool_call_response_to_function_call(tool_resp)
        if not second_fc:
            print(f"Warning: Could not convert tool response to function call for query: {query[:50]}...")
            continue
        '''
        notice:
        [
        {
            "conversations": [
            {
                "from": "human",
                "value": "user instruction"
            },
            {
                "from": "function_call",
                "value": "tool arguments"
            },
            {
                "from": "observation",
                "value": "tool result"
            },
            {
                "from": "gpt",
                "value": "model response"
            }
            ],
            "system": "system prompt (optional)",
            "tools": "tool description (optional)"
        }
        ]
        Note that the human and observation should appear in odd positions, while gpt and function should appear in even positions. The gpt and function will be learned by the model.
        偶数轮的信息才能被学习到，奇数轮的信息不能被学习到。
        '''
        # 6. 构建完整的对话数据
        conversation = [
            {
                "from": "human", 
                "value": query
            },
            {
                "from": "function_call",
                "value": first_fc
            },
            {
                "from": "observation",
                "value": json.dumps(retrieval_resp.get("response", {}).get("result", []), ensure_ascii=False)
            },
            {
                "from": "function_call",
                "value": second_fc
            },
            {
                "from": "observation", 
                "value": extract_tool_response_result(tool_resp)
            },
            {
                "from": "gpt",
                "value": final_answer
            }
        ]
        
        # 7. 添加到结果中
        converted_item = {
            "conversations": conversation,
            "system": build_system_message(),
            "tools": json.dumps(tools_definition, ensure_ascii=False)
        }
        
        converted_data.append(converted_item)
    
    # 过滤数据：移除第一步observation为空或final_answer为默认回答的数据
    print(f"Filtering data...")
    original_count = len(converted_data)
    filtered_data = []
    
    for item in converted_data:
        conversations = item.get("conversations", [])
        
        # 检查第一个observation是否为空
        first_observation = None
        final_answer = None
        
        for conv in conversations:
            if conv.get("from") == "observation" and first_observation is None:
                first_observation = conv.get("value", "")
            elif conv.get("from") == "gpt":
                final_answer = conv.get("value", "")
        
        # 过滤条件：
        # 1. 第一个observation不是"[]"
        # 2. final_answer不是默认回答
        if (first_observation != "[]" and 
            first_observation != "[]" and 
            final_answer != "抱歉，暂时无法提供详细分析。"):
            filtered_data.append(item)
        else:
            print(f"Filtered out item with empty observation or default answer")
    
    print(f"Filtered {original_count - len(filtered_data)} items, kept {len(filtered_data)} items")
    
    # 分割训练集和测试集
    print(f"Splitting {len(filtered_data)} items into train/test sets with ratio {train_ratio}")
    train_data, test_data = split_train_test(filtered_data, train_ratio, random_seed)
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存训练集
    train_path = Path(output_dir) / "function_call_train.json"
    print(f"Saving {len(train_data)} training items to {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    test_path = Path(output_dir) / "function_call_test.json"
    print(f"Saving {len(test_data)} test items to {test_path}")
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 保存完整数据集
    full_path = Path(output_dir) / "function_call_full.json"
    print(f"Saving {len(filtered_data)} full items to {full_path}")
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion completed!")
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print(f"Total: {len(filtered_data)} samples")
    print(f"Filtered out: {original_count - len(filtered_data)} samples")


def main():
    """主函数"""
    # 文件路径配置
    base_path = "/home/ziqiang/LLaMA-Factory/data/dataset/9_10"
    
    retrieval_fc_path = f"{base_path}/retrieval_tool_toolcall_evaluate_topk5.json"
    retrieval_response_path = f"{base_path}/retrieval_evaluate_topk5_raw.json"
    tool_response_path = f"{base_path}/9.5_evaluate_tool_call_responses.json"
    answer_path = f"{base_path}/9.5_evaluate_answer_data.cleaned.json"
    output_dir = "/home/ziqiang/LLaMA-Factory/data/dataset/9_10/function_call_data"
    train_ratio = 1
    random_seed = 42
    
    # 检查文件是否存在
    for path in [retrieval_fc_path, retrieval_response_path, tool_response_path, answer_path]:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    print(f"Input directory: {base_path}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: {train_ratio}")
    print(f"Random seed: {random_seed}")
    
    # 执行转换
    convert_to_function_call_format(
        retrieval_fc_path,
        retrieval_response_path,
        tool_response_path, 
        answer_path,
        output_dir,
        train_ratio,
        random_seed
    )


if __name__ == "__main__":
    main()
