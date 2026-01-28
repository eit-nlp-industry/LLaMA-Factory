#!/usr/bin/env python3
"""
将JSON文件中retrieval_tool的function_call从完整JSON格式简化为只保留query和source_filter
"""

import json
import sys
import os
import re


def simplify_retrieval_tool_call(value_str):
    """简化retrieval_tool的函数调用格式"""
    try:
        # 尝试解析JSON
        call_data = json.loads(value_str)
        
        # 检查是否是retrieval_tool调用
        if isinstance(call_data, dict) and call_data.get('name') == 'retrieval_tool':
            arguments = call_data.get('arguments', {})
            query = arguments.get('query', '')
            source_filter = arguments.get('source_filter', '')
            
            # 返回简化格式
            return f'"query": "{query}", "source_filter": "{source_filter}"'
        
        # 如果不是retrieval_tool，返回原值
        return value_str
    except json.JSONDecodeError:
        # 如果不是有效的JSON，返回原值
        return value_str


def process_conversations(conversations):
    """处理conversations数组中的function_call"""
    if not isinstance(conversations, list):
        return
    
    for conv in conversations:
        if isinstance(conv, dict):
            # 如果是function_call类型，检查并简化
            if conv.get('from') == 'function_call' and 'value' in conv:
                old_value = conv['value']
                new_value = simplify_retrieval_tool_call(old_value)
                if new_value != old_value:
                    conv['value'] = new_value
                    print(f"  - 简化: {old_value[:100]}... -> {new_value[:100]}...")


def process_data(data):
    """递归处理数据"""
    if isinstance(data, dict):
        # 检查是否有conversations字段
        if 'conversations' in data:
            process_conversations(data['conversations'])
        
        # 递归处理字典中的所有值
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                process_data(value)
    
    elif isinstance(data, list):
        # 如果是列表，处理每个元素
        for item in data:
            process_data(item)


def process_json_file(file_path):
    """处理JSON文件"""
    print(f"处理文件: {file_path}")
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    process_data(data)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已完成: {file_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python simplify_retrieval_tool_calls.py <json_file1> [json_file2] ...")
        print("或: python simplify_retrieval_tool_calls.py <directory>")
        sys.exit(1)
    
    paths = sys.argv[1:]
    
    for path in paths:
        # 转换为绝对路径
        abs_path = os.path.abspath(path)
        
        if os.path.isfile(abs_path) and abs_path.endswith('.json'):
            process_json_file(abs_path)
        elif os.path.isdir(abs_path):
            # 如果是目录，处理目录下所有JSON文件
            for root, dirs, files in os.walk(abs_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        process_json_file(file_path)
        else:
            print(f"✗ 文件或目录不存在: {path}")


if __name__ == '__main__':
    main()

