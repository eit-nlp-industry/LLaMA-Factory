#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为训练数据添加工具调用失败的场景
当第二次function_call失败时，直接返回失败提示
"""

import json
import copy
from pathlib import Path


def create_failure_case(original_data):
    """
    为一条对话数据创建失败场景版本
    
    在第二次function_call的位置失败，删除第二次function_call及其后续的observation和gpt，
    直接用失败消息替代
    """
    # 深拷贝原始数据
    failure_data = copy.deepcopy(original_data)
    
    conversations = failure_data["conversations"]
    
    # 找到第二次function_call的索引
    function_call_count = 0
    second_function_call_index = -1
    
    for i, conv in enumerate(conversations):
        if conv["from"] == "function_call":
            function_call_count += 1
            if function_call_count == 2:
                second_function_call_index = i
                break
    
    # 如果没有第二次function_call，返回None
    if second_function_call_index == -1:
        return None
    
    # 保留到第二次function_call之前的所有内容（human -> function_call -> observation）
    # 也就是保留前3个元素
    failure_data["conversations"] = conversations[:second_function_call_index]
    
    # 添加失败的gpt消息
    failure_data["conversations"].append({
        "from": "gpt",
        "value": "工具检索失败，请您稍后再试或者联系业务人员。"
    })
    
    return failure_data


def process_training_data(input_file, output_file):
    """
    处理训练数据，为每条有多次function_call的数据创建失败版本
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"原始数据共有 {len(original_data)} 条")
    
    # 统计有第二次function_call的数据
    multi_call_count = 0
    for item in original_data:
        function_call_count = sum(1 for conv in item["conversations"] if conv["from"] == "function_call")
        if function_call_count >= 2:
            multi_call_count += 1
    
    print(f"其中有 {multi_call_count} 条数据包含2次或以上的function_call")
    
    # 创建新数据集（包含原始数据和失败场景数据）
    new_dataset = []
    failure_cases_added = 0
    
    for item in original_data:
        # 添加原始数据
        new_dataset.append(item)
        
        # 创建失败场景
        failure_case = create_failure_case(item)
        if failure_case:
            new_dataset.append(failure_case)
            failure_cases_added += 1
    
    print(f"成功添加 {failure_cases_added} 条失败场景数据")
    print(f"新数据集共有 {len(new_dataset)} 条")
    
    # 保存新数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到: {output_file}")
    
    return new_dataset


def main():
    input_file = "/home/ziqiang/LLaMA-Factory/data/dataset/11_10/fuzzy_result_20251112_235626_final_completed.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/dataset/11_10/fuzzy_result_20251112_235626_final_completed_with_failures.json"
    
    print("=" * 60)
    print("开始处理训练数据，添加工具调用失败场景")
    print("=" * 60)
    
    process_training_data(input_file, output_file)
    
    print("=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

