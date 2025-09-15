#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据合并和格式转换工具
支持合并多种格式的JSON文件并转换为统一格式用于LLaMA-Factory训练
"""

import json
import os
import argparse
from typing import List, Dict, Any, Union
import glob
from datetime import datetime

class DataMerger:
    def __init__(self):
        self.merged_data = []
        
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载文件: {file_path}, 包含 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            return []
    
    def convert_custom_context_to_sharegpt(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """将custom_context格式转换为sharegpt格式"""
        conversations = [
            {
                "from": "human",
                "value": f"{item['instruction']}\n\n输入文本：\n{item['input']}"
            },
            {
                "from": "gpt", 
                "value": item['output']
            }
        ]
        
        return {
            "conversations": conversations,
            "system": "你是一个专业的信息抽取助手，能够从OCR文本中准确提取指定的信息。",
            "context": item.get('context', {})
        }
    
    def is_sharegpt_format(self, item: Dict[str, Any]) -> bool:
        """判断是否为sharegpt格式"""
        return 'conversations' in item and isinstance(item['conversations'], list)
    
    def is_custom_context_format(self, item: Dict[str, Any]) -> bool:
        """判断是否为custom_context格式"""
        return all(key in item for key in ['instruction', 'input', 'output'])
    
    def merge_files(self, file_paths: List[str], output_path: str, 
                   convert_to_sharegpt: bool = True) -> None:
        """合并多个JSON文件"""
        total_records = 0
        converted_count = 0
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 {file_path}")
                continue
                
            data = self.load_json_file(file_path)
            if not data:
                continue
                
            for item in data:
                if convert_to_sharegpt:
                    if self.is_sharegpt_format(item):
                        # 已经是sharegpt格式，直接添加
                        self.merged_data.append(item)
                    elif self.is_custom_context_format(item):
                        # 转换为sharegpt格式
                        converted_item = self.convert_custom_context_to_sharegpt(item)
                        self.merged_data.append(converted_item)
                        converted_count += 1
                    else:
                        print(f"警告: 未知格式的数据项，跳过")
                        continue
                else:
                    # 保持原格式
                    self.merged_data.append(item)
                
                total_records += 1
        
        # 保存合并后的数据
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.merged_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n=== 合并完成 ===")
            print(f"成功合并 {len(file_paths)} 个文件到 {output_path}")
            print(f"总共包含 {total_records} 条记录")
            if convert_to_sharegpt:
                print(f"其中转换了 {converted_count} 条custom_context格式的记录")
        except Exception as e:
            print(f"保存合并文件时出错: {e}")
    
    def create_dataset_info_entry(self, dataset_name: str, file_path: str, 
                                formatting: str = "sharegpt") -> Dict[str, Any]:
        """创建dataset_info.json的条目"""
        if formatting == "sharegpt":
            return {
                dataset_name: {
                    "file_name": file_path,
                    "formatting": "sharegpt",
                    "columns": {
                        "messages": "conversations",
                        "system": "system",
                        "tools": "tools"
                    }
                }
            }
        else:
            return {
                dataset_name: {
                    "file_name": file_path,
                    "formatting": formatting
                }
            }

def main():
    parser = argparse.ArgumentParser(description='合并和转换多种格式的JSON文件')
    parser.add_argument('--input', '-i', nargs='+', required=True, 
                       help='输入文件路径列表')
    parser.add_argument('--output', '-o', required=True,
                       help='输出文件路径')
    parser.add_argument('--pattern', '-p', 
                       help='使用glob模式匹配文件 (例如: "*.json")')
    parser.add_argument('--convert-to-sharegpt', action='store_true', default=True,
                       help='将所有格式转换为sharegpt格式 (默认: True)')
    parser.add_argument('--dataset-name', 
                       help='数据集名称，用于生成dataset_info.json条目')
    parser.add_argument('--update-dataset-info', action='store_true',
                       help='更新dataset_info.json文件')
    
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
    
    # 执行合并
    merger = DataMerger()
    merger.merge_files(file_paths, args.output, args.convert_to_sharegpt)
    
    # 生成dataset_info.json条目
    if args.dataset_name:
        entry = merger.create_dataset_info_entry(
            args.dataset_name, 
            args.output, 
            "sharegpt" if args.convert_to_sharegpt else "custom_context"
        )
        
        print(f"\n=== Dataset Info 条目 ===")
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        
        # 更新dataset_info.json文件
        if args.update_dataset_info:
            dataset_info_path = "/home/ziqiang/LLaMA-Factory/data/dataset_info.json"
            try:
                with open(dataset_info_path, 'r', encoding='utf-8') as f:
                    dataset_info = json.load(f)
                
                dataset_info.update(entry)
                
                with open(dataset_info_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_info, f, ensure_ascii=False, indent=2)
                
                print(f"已更新 {dataset_info_path}")
            except Exception as e:
                print(f"更新dataset_info.json时出错: {e}")

if __name__ == "__main__":
    main()
    '''
    python /home/ziqiang/LLaMA-Factory/data/dataset/merge_and_convert_data.py \
    --input /home/ziqiang/LLaMA-Factory/data/dataset/9_14/function_call_train.json /home/ziqiang/LLaMA-Factory/data/dataset/9_14/text_idcards_8_18.json /home/ziqiang/LLaMA-Factory/data/dataset/9_14/price_cal_train_09_14.json /home/ziqiang/LLaMA-Factory/data/dataset/9_14/price_policy_train_09_14.json\
    --output /home/ziqiang/LLaMA-Factory/data/dataset/9_14/mixed_training_data.json \
    --convert-to-sharegpt \
    --dataset-name mixed_training_data \
    --update-dataset-info
    '''