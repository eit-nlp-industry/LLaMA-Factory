#!/usr/bin/env python3
"""
将训练数据从 context_time 格式转换为 context 对象格式
"""

import json
import re
import os
from typing import Dict, Any

def convert_time_format(time_str: str) -> str:
    """
    将时间字符串转换为 YYYYMMDD 格式
    
    Args:
        time_str: 输入的时间字符串（如"7月16日"）
    
    Returns:
        YYYYMMDD格式的时间字符串（如"20250716"）
    """
    # 如果已经是8位数字格式，直接返回
    if re.match(r'^\d{8}$', time_str):
        return time_str
        
    # 处理 "7月16日" 格式
    month_day_match = re.search(r'(\d{1,2})月(\d{1,2})日', time_str)
    if month_day_match:
        month = int(month_day_match.group(1))
        day = int(month_day_match.group(2))
        return f"2025{month:02d}{day:02d}"
        
    # 处理 "7.16" 格式
    dot_format_match = re.search(r'(\d{1,2})\.(\d{1,2})', time_str)
    if dot_format_match:
        month = int(dot_format_match.group(1))
        day = int(dot_format_match.group(2))
        return f"2025{month:02d}{day:02d}"
        
    # 处理 "2025-07-16" 格式
    iso_format_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', time_str)
    if iso_format_match:
        year = iso_format_match.group(1)
        month = int(iso_format_match.group(2))
        day = int(iso_format_match.group(3))
        return f"{year}{month:02d}{day:02d}"
        
    # 如果无法解析，返回默认值
    return "20250101"

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换单个训练样本
    
    Args:
        sample: 原始训练样本
    
    Returns:
        转换后的训练样本
    """
    converted_sample = sample.copy()
    
    # 如果存在 context_time 字段，转换为 context 对象
    if 'context_time' in sample:
        time_formatted = convert_time_format(sample['context_time'])
        converted_sample['context'] = {"time": time_formatted}
        del converted_sample['context_time']
    
    return converted_sample

def convert_dataset(input_file: str, output_file: str):
    """
    转换整个数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在转换数据集: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for i, sample in enumerate(data):
        converted_sample = convert_sample(sample)
        converted_data.append(converted_sample)
        
        # 显示进度
        if (i + 1) % 50 == 0 or i == 0:
            print(f"已转换 {i + 1}/{len(data)} 个样本")
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共转换 {len(converted_data)} 个样本")
    print(f"输出文件: {output_file}")

def show_sample_comparison(input_file: str, sample_index: int = 0):
    """
    显示转换前后的样本对比
    
    Args:
        input_file: 输入文件路径
        sample_index: 要显示的样本索引
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if sample_index >= len(data):
        print(f"样本索引 {sample_index} 超出范围，数据集共有 {len(data)} 个样本")
        return
    
    original_sample = data[sample_index]
    converted_sample = convert_sample(original_sample)
    
    print("="*80)
    print("转换前后对比:")
    print("="*80)
    
    print("转换前:")
    print(json.dumps(original_sample, ensure_ascii=False, indent=2))
    
    print("\n" + "-"*80 + "\n")
    
    print("转换后:")
    print(json.dumps(converted_sample, ensure_ascii=False, indent=2))
    
    print("="*80)

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "/home/ziqiang/LLaMA-Factory/data/ocr_text_orders_30_updated_with_context.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/ocr_text_orders_30_updated_with_context_object.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        exit(1)
    
    # 显示样本对比
    print("显示第一个样本的转换对比:")
    show_sample_comparison(input_file, 0)
    
    print("\n" + "="*100 + "\n")
    
    # 执行转换
    convert_dataset(input_file, output_file)
