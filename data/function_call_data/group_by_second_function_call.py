#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据第二次function_call中的工具名称对query进行分组
如果没有第二次function_call，则过滤掉该记录
只保存第一次function_call中的query，而不是整个JSON数据
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

def extract_first_function_call_query(conversations):
    """
    从conversations中提取第一次function_call的query
    返回query字符串，如果没有第一次function_call或无法提取则返回None
    """
    function_calls = [item for item in conversations if item.get("from") == "function_call"]
    
    # 如果没有至少1个function_call，返回None
    if len(function_calls) < 1:
        return None
    
    # 获取第一次function_call
    first_function_call = function_calls[0]
    value = first_function_call.get("value", "")
    
    # 提取query字段
    # 格式可能是: "query": "查询内容", "source_filter": "toollist"
    # 或者可能是JSON格式
    if isinstance(value, dict):
        return value.get("query")
    
    if not isinstance(value, str):
        return None
    
    # 尝试多种方法提取query
    # 方法1: 如果value是JSON格式，直接解析
    value_stripped = value.strip()
    if value_stripped.startswith('{'):
        try:
            function_data = json.loads(value)
            if isinstance(function_data, dict):
                return function_data.get("query")
        except json.JSONDecodeError:
            pass
    
    # 方法2: 如果value是转义的JSON字符串（如 "\"query\": \"...\""），先去掉外层引号
    if value_stripped.startswith('"') and value_stripped.endswith('"'):
        try:
            # 尝试去掉外层引号并解析
            unescaped = json.loads(value)
            if isinstance(unescaped, str):
                # 如果解析后还是字符串，可能是转义的JSON
                try:
                    function_data = json.loads(unescaped)
                    if isinstance(function_data, dict):
                        return function_data.get("query")
                except (json.JSONDecodeError, TypeError):
                    pass
        except json.JSONDecodeError:
            pass
    
    # 方法3: 使用正则表达式提取query
    # 匹配 "query": "..." 格式，处理转义的双引号
    match = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)"', value)
    if match:
        # 处理转义字符
        query = match.group(1).replace('\\"', '"').replace('\\\\', '\\')
        return query
    
    # 尝试匹配 'query': '...' 格式
    match = re.search(r"'query'\s*:\s*'((?:[^'\\]|\\.)*)'", value)
    if match:
        query = match.group(1).replace("\\'", "'").replace('\\\\', '\\')
        return query
    
    return None

def extract_second_function_call_name(conversations):
    """
    从conversations中提取第二次function_call的工具名称
    返回工具名称，如果没有第二次function_call则返回None
    """
    function_calls = [item for item in conversations if item.get("from") == "function_call"]
    
    # 如果没有至少2个function_call，返回None
    if len(function_calls) < 2:
        return None
    
    # 获取第二次function_call
    second_function_call = function_calls[1]
    value = second_function_call.get("value", "")
    
    # 尝试解析JSON字符串
    try:
        # 如果value是JSON字符串，解析它
        if isinstance(value, str):
            # 处理可能的转义字符
            value = value.strip()
            # 尝试解析JSON
            function_data = json.loads(value)
            if isinstance(function_data, dict):
                return function_data.get("name")
            # 如果解析后不是字典，可能是其他格式
            return None
        elif isinstance(value, dict):
            # 如果已经是字典，直接获取name
            return value.get("name")
    except (json.JSONDecodeError, AttributeError) as e:
        # 如果解析失败，尝试其他方法
        # 有些function_call的value可能是特殊格式，如retrieval_tool
        # 检查是否是retrieval_tool格式（非JSON格式）
        if value.startswith('"query"'):
            return None  # retrieval_tool通常没有name字段，跳过
        print(f"警告: 无法解析function_call的value: {value[:100]}... 错误: {e}")
        return None
    
    return None

def main():
    input_file = "/home/ziqiang/LLaMA-Factory/data/dataset/11_10/11_14_train_data_filtered_humanized_rewritten.json"
    output_dir = "/home/ziqiang/LLaMA-Factory/data/dataset/11_10/grouped_by_tool"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取输入文件
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共 {len(data)} 条记录")
    
    # 按工具名称分组，只保存query
    grouped_queries = defaultdict(list)
    filtered_count = 0
    no_query_count = 0
    
    for idx, item in enumerate(data):
        conversations = item.get("conversations", [])
        
        # 提取第二次function_call的工具名称
        tool_name = extract_second_function_call_name(conversations)
        if tool_name is None:
            filtered_count += 1
            continue
        
        # 提取第一次function_call的query
        query = extract_first_function_call_query(conversations)
        if query is None:
            no_query_count += 1
            continue
        
        grouped_queries[tool_name].append(query)
    
    print(f"过滤掉 {filtered_count} 条没有第二次function_call的记录")
    print(f"过滤掉 {no_query_count} 条无法提取query的记录")
    print(f"找到 {len(grouped_queries)} 个不同的工具")
    
    # 为每个工具创建JSON文件，只保存query列表
    for tool_name, queries in grouped_queries.items():
        # 清理工具名称，用于文件名（移除特殊字符）
        safe_tool_name = "".join(c for c in tool_name if c.isalnum() or c in ('-', '_'))
        output_file = os.path.join(output_dir, f"{safe_tool_name}.json")
        
        print(f"正在写入 {tool_name}: {len(queries)} 条query -> {output_file}")
        
        # 只保存query列表
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    for tool_name, queries in sorted(grouped_queries.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{tool_name}: {len(queries)} 条query")
    
    print(f"\n所有文件已保存到: {output_dir}")

if __name__ == "__main__":
    main()

