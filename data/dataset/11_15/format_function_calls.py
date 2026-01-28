#!/usr/bin/env python3
"""
脚本功能：将JSON文件中第二轮的function_call从紧凑格式转换为格式化版本

使用方法：
    python format_function_calls.py <input_file> [output_file]
    
如果不指定output_file，会直接修改原文件（建议先备份）
"""

import json
import sys
import re
from pathlib import Path


def is_compact_format(value_str):
    """检查是否是紧凑格式（冒号后没有空格）"""
    # 在JSON解析后，字符串中的转义已经被解析
    # 检查是否包含紧凑格式的模式："name":"（冒号后直接是引号，没有空格）
    if re.search(r'"name":\s*"[^"]', value_str) and not re.search(r'"name":\s+"[^"]', value_str):
        return True
    # 检查是否包含格式化版本："name": "（冒号后有空格）
    if re.search(r'"name":\s+"[^"]', value_str):
        return False
    # 如果都没有匹配，检查是否有"name":的模式
    if '"name":' in value_str:
        # 检查冒号后是否有空格
        match = re.search(r'"name":(\s*)"', value_str)
        if match and not match.group(1):  # 没有空格
            return True
    return False


def is_tool_call_format(value_str):
    """检查是否是工具调用格式（不是retrieval_tool格式）"""
    # retrieval_tool格式通常以 "query": 开头
    if value_str.strip().startswith('"query":'):
        return False
    # 工具调用格式以 {"name": 开头（JSON解析后已经是真正的引号）
    if value_str.strip().startswith('{"name"'):
        return True
    return False


def format_json_string(compact_json_str):
    """将紧凑格式的JSON字符串转换为格式化版本"""
    try:
        # 在JSON解析后，value_str已经是解析过的字符串，可以直接解析
        parsed = json.loads(compact_json_str)
        
        # 格式化为字符串，使用空格作为分隔符（逗号后和冒号后都有空格）
        formatted = json.dumps(parsed, ensure_ascii=False, separators=(', ', ': '))
        
        return formatted
    except (json.JSONDecodeError, Exception) as e:
        print(f"警告：解析JSON失败: {e}")
        print(f"原始字符串: {compact_json_str[:100]}...")
        return compact_json_str


def find_second_round_function_calls(conversations):
    """找到第二轮的function_call项（工具调用格式，非retrieval_tool）"""
    function_calls = []
    for i, conv in enumerate(conversations):
        if conv.get("from") == "function_call":
            value = conv.get("value", "")
            # 检查是否是工具调用格式（不是retrieval_tool）
            if is_tool_call_format(value):
                function_calls.append(i)
    return function_calls


def process_data(data_list):
    """处理所有data项"""
    modified_count = 0
    total_second_round_calls = 0
    
    for data_idx, data in enumerate(data_list):
        conversations = data.get("conversations", [])
        if not conversations:
            continue
            
        # 找到第二轮的function_call
        second_round_indices = find_second_round_function_calls(conversations)
        total_second_round_calls += len(second_round_indices)
        
        for idx in second_round_indices:
            conv = conversations[idx]
            value = conv.get("value", "")
            
            # 检查是否是紧凑格式
            if is_compact_format(value):
                # 转换为格式化版本
                formatted_value = format_json_string(value)
                if formatted_value != value:
                    conv["value"] = formatted_value
                    modified_count += 1
                    print(f"Data {data_idx + 1}, conversation {idx + 1}: 已格式化")
    
    return modified_count, total_second_round_calls


def main():
    if len(sys.argv) < 2:
        print("用法: python format_function_calls.py <input_file> [output_file]")
        print("如果不指定output_file，会直接修改原文件")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"错误：文件不存在: {input_file}")
        sys.exit(1)
    
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file
    
    # 读取JSON文件
    print(f"正在读取文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"错误：读取JSON文件失败: {e}")
        sys.exit(1)
    
    if not isinstance(data_list, list):
        print("错误：JSON文件应该包含一个数组")
        sys.exit(1)
    
    print(f"找到 {len(data_list)} 条数据")
    
    # 处理数据
    modified_count, total_second_round_calls = process_data(data_list)
    
    print(f"\n处理完成:")
    print(f"  总共找到 {total_second_round_calls} 个第二轮的function_call")
    print(f"  其中 {modified_count} 个从紧凑格式转换为格式化版本")
    
    # 保存文件
    print(f"\n正在保存到: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print("保存成功！")
    except Exception as e:
        print(f"错误：保存文件失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

