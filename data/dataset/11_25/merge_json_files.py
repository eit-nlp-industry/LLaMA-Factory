#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并多个JSON文件为一个JSON文件的脚本
将指定目录下所有JSON文件的内容合并成一个数组
"""

import json
import os
import glob
from pathlib import Path


def merge_json_files(input_dir, output_file, exclude_patterns=None):
    """
    合并指定目录下所有JSON文件为一个文件
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出文件路径
        exclude_patterns: 要排除的文件名模式列表（例如：['merge_json_files.py', 'count_tokens.py']）
    """
    if exclude_patterns is None:
        exclude_patterns = ['merge_json_files.py', 'count_tokens.py']
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    
    # 过滤掉输出文件本身和排除的文件
    if output_file in json_files:
        json_files.remove(output_file)
    
    # 过滤排除的文件名模式
    json_files = [f for f in json_files if not any(
        pattern in os.path.basename(f) for pattern in exclude_patterns
    )]
    
    if not json_files:
        print(f"警告: 在目录 {input_dir} 中没有找到JSON文件")
        return
    
    # 合并所有JSON数据
    merged_data = []
    
    for json_file in sorted(json_files):  # 按文件名排序
        try:
            print(f"正在处理: {os.path.basename(json_file)}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 如果数据是列表，则扩展；如果是字典，则追加
                if isinstance(data, list):
                    merged_data.extend(data)
                    print(f"  - 添加了 {len(data)} 条记录")
                elif isinstance(data, dict):
                    merged_data.append(data)
                    print(f"  - 添加了 1 条记录")
                else:
                    print(f"  - 警告: 文件包含非列表/字典数据，已跳过")
                    
        except json.JSONDecodeError as e:
            print(f"  - 错误: 无法解析JSON文件 {json_file}: {e}")
        except Exception as e:
            print(f"  - 错误: 处理文件 {json_file} 时出错: {e}")
    
    # 保存合并后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n合并完成!")
        print(f"共处理 {len(json_files)} 个JSON文件")
        print(f"合并后共有 {len(merged_data)} 条记录")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"\n错误: 无法保存合并后的文件: {e}")


if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = "/home/ziqiang/LLaMA-Factory/data/dataset/11_25"
    # 设置输入目录和输出文件
    input_directory = script_dir
    output_filename = os.path.join(script_dir, 'price_service.json')
    
    # 执行合并
    print(f"输入目录: {input_directory}")
    print(f"输出文件: {output_filename}\n")
    
    merge_json_files(input_directory, output_filename)
