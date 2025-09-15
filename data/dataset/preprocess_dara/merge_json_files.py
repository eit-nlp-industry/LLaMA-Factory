#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON文件合并工具
支持合并多个JSON文件到一个文件中
"""

import json
import os
import argparse
from typing import List, Dict, Any
import glob

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载文件: {file_path}, 包含 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return []

def merge_json_files(file_paths: List[str], output_path: str) -> None:
    """合并多个JSON文件"""
    merged_data = []
    total_records = 0
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            data = load_json_file(file_path)
            if data:
                merged_data.extend(data)
                total_records += len(data)
        else:
            print(f"警告: 文件不存在 {file_path}")
    
    # 保存合并后的数据
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"成功合并 {len(file_paths)} 个文件到 {output_path}")
        print(f"总共包含 {total_records} 条记录")
    except Exception as e:
        print(f"保存合并文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='合并多个JSON文件')
    parser.add_argument('--input', '-i', nargs='+', required=True, 
                       help='输入文件路径列表')
    parser.add_argument('--output', '-o', required=True,
                       help='输出文件路径')
    parser.add_argument('--pattern', '-p', 
                       help='使用glob模式匹配文件 (例如: "*.json")')
    
    args = parser.parse_args()
    
    file_paths = []
    
    if args.pattern:
        # 使用glob模式匹配
        file_paths = glob.glob(args.pattern)
        print(f"使用模式 '{args.pattern}' 找到 {len(file_paths)} 个文件")
    else:
        # 使用指定的文件列表
        file_paths = args.input
    
    if not file_paths:
        print("没有找到要合并的文件")
        return
    
    merge_json_files(file_paths, args.output)

if __name__ == "__main__":
    main()
    '''
    python /home/ziqiang/LLaMA-Factory/data/dataset/merge_json_files.py --input /home/ziqiang/LLaMA-Factory/data/dataset/9_14/text_idcards_8_18_eval.json /home/ziqiang/LLaMA-Factory/data/dataset/9_14/price_cal_test_09_14.json /home/ziqiang/LLaMA-Factory/data/dataset/9_14/price_policy_test_09_14.json --output /home/ziqiang/LLaMA-Factory/data/dataset/9_14/merged_dataset.json
    '''