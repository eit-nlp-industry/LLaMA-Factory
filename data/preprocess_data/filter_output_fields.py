#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选JSON/JSONL文件中output字段包含特定关键字的条目
"""

import json
import argparse
from pathlib import Path

def read_json_or_jsonl(file_path):
    """
    读取JSON或JSONL文件
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        list: 数据列表
    """
    data = []
    file_ext = Path(file_path).suffix.lower()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_ext == '.jsonl':
            # 读取JSONL文件
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 解析行时出错 - {e}")
                        continue
        else:
            # 读取JSON文件
            data = json.load(f)
    
    return data

def write_json_or_jsonl(data, file_path):
    """
    写入JSON或JSONL文件
    
    Args:
        data (list): 要写入的数据
        file_path (str): 输出文件路径
    """
    file_ext = Path(file_path).suffix.lower()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_ext == '.jsonl':
            # 写入JSONL文件
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            # 写入JSON文件
            json.dump(data, f, ensure_ascii=False, indent=2)

def filter_json_by_output_fields(input_file, output_file, target_fields):
    """
    筛选JSON/JSONL文件中output字段包含指定关键字的条目
    
    Args:
        input_file (str): 输入JSON/JSONL文件路径
        output_file (str): 输出JSON/JSONL文件路径  
        target_fields (list): 要筛选的字段列表
    """
    
    # 读取原始文件
    data = read_json_or_jsonl(input_file)
    
    print(f"原始数据总条数: {len(data)}")
    print(f"输入文件格式: {Path(input_file).suffix}")
    
    # 筛选包含目标字段的条目
    filtered_data = []
    
    for item in data:
        if 'output' in item:
            output_str = item['output']
            # 检查output字段是否包含任何目标字段
            if any(field in output_str for field in target_fields):
                filtered_data.append(item)
    
    print(f"筛选后数据条数: {len(filtered_data)}")
    
    # 按目标字段分类统计
    field_counts = {}
    for field in target_fields:
        count = sum(1 for item in filtered_data if field in item['output'])
        field_counts[field] = count
        print(f"包含 '{field}' 的条目数: {count}")
    
    # 如果没有指定输出文件扩展名，使用与输入文件相同的格式
    if not Path(output_file).suffix:
        input_ext = Path(input_file).suffix
        output_file = output_file + input_ext
    
    # 保存筛选结果
    write_json_or_jsonl(filtered_data, output_file)
    
    print(f"筛选结果已保存到: {output_file}")
    print(f"输出文件格式: {Path(output_file).suffix}")
    
    return filtered_data, field_counts

def main():
    parser = argparse.ArgumentParser(description='筛选JSON/JSONL文件中包含特定字段的条目')
    parser.add_argument('input_file', help='输入JSON/JSONL文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（会自动保持与输入文件相同的格式）', 
                       default='filtered_output')
    parser.add_argument('-f', '--fields', nargs='+', 
                       default=['resource_detail', 'team_size', 'resource_team_size','end_date','start_date','resource_end_time','resource_start_time','contacts'],
                       help='要筛选的字段列表')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_file).exists():
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        return
    
    # 执行筛选
    try:
        filtered_data, field_counts = filter_json_by_output_fields(
            args.input_file, args.output, args.fields
        )
        
        print("\n筛选完成!")
        print(f"目标字段: {args.fields}")
        print(f"输出文件: {args.output}")
        
    except Exception as e:
        print(f"筛选过程中出现错误: {e}")

if __name__ == "__main__":
    main()
