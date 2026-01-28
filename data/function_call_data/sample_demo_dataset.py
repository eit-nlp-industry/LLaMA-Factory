#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练数据中按工具名称分类采样，生成demo训练集
"""

import json
from collections import defaultdict
import random

def extract_second_function_call_tool(conversations):
    """提取第二轮function_call的工具名称"""
    function_calls = []
    for msg in conversations:
        if msg.get("from") == "function_call":
            function_calls.append(msg)
    
    # 如果有第二个function_call，提取其工具名称
    if len(function_calls) >= 2:
        try:
            second_call = json.loads(function_calls[1]["value"])
            return second_call.get("name")
        except:
            return None
    return None

def main():
    # 读取原始数据
    input_file = "/home/ziqiang/LLaMA-Factory/data/dataset/10_22/10.22_train_data.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/dataset/10_22/demo_dataset_sampled.json"
    
    print("正在读取原始数据文件...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}")
    
    # 按工具名称分组
    tool_groups = defaultdict(list)
    no_tool_count = 0
    
    for idx, item in enumerate(data):
        tool_name = extract_second_function_call_tool(item.get("conversations", []))
        if tool_name:
            tool_groups[tool_name].append(item)
        else:
            no_tool_count += 1
    
    print(f"\n找到的工具及其数据量:")
    for tool_name, items in sorted(tool_groups.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {tool_name}: {len(items)} 条")
    print(f"  未找到第二轮function_call: {no_tool_count} 条")
    
    # 每个工具采样数据
    samples_per_tool = 3  # 每个工具采样3条
    sampled_data = []
    
    print(f"\n开始采样（每个工具采样 {samples_per_tool} 条）...")
    for tool_name, items in sorted(tool_groups.items()):
        # 如果该工具的数据少于采样数，就全部取出
        n_samples = min(samples_per_tool, len(items))
        sampled = random.sample(items, n_samples)
        sampled_data.extend(sampled)
        print(f"  {tool_name}: 采样了 {n_samples} 条")
    
    print(f"\n总采样数据量: {len(sampled_data)}")
    
    # 保存采样后的数据
    print(f"正在保存到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print("完成！")
    
    # 输出统计信息
    print(f"\n采样结果统计:")
    print(f"  工具种类数: {len(tool_groups)}")
    print(f"  demo数据集大小: {len(sampled_data)}")
    
    # 再次统计demo数据集中的工具分布
    demo_tool_count = defaultdict(int)
    for item in sampled_data:
        tool_name = extract_second_function_call_tool(item.get("conversations", []))
        if tool_name:
            demo_tool_count[tool_name] += 1
    
    print(f"\ndemo数据集中的工具分布:")
    for tool_name, count in sorted(demo_tool_count.items()):
        print(f"  {tool_name}: {count} 条")

if __name__ == "__main__":
    main()

