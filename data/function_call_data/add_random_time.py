#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机时间添加脚本
功能：为JSON数据中的每条记录添加或重置随机时间字段
时间格式：YYYY-MM-DD HH:MM
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import sys


def generate_random_time(start_date="2025-09-01", end_date="2025-10-31"):
    """
    生成随机时间字符串
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        str: 格式化的时间字符串 "YYYY-MM-DD HH:MM"
    """
    # 解析日期
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 计算日期范围
    time_delta = end - start
    random_days = random.randint(0, time_delta.days)
    
    # 随机小时和分钟
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    
    # 生成随机时间
    random_date = start + timedelta(
        days=random_days,
        hours=random_hours,
        minutes=random_minutes
    )
    
    # 返回格式化的时间字符串
    return random_date.strftime("%Y-%m-%d %H:%M")


def add_time_to_json(input_file, output_file=None, force_reset=False, 
                     start_date="2025-09-01", end_date="2025-10-31"):
    """
    为JSON文件中的每条记录添加或重置时间字段
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径，如果为None则覆盖原文件
        force_reset: 是否强制重置已有的时间字段
        start_date: 随机时间的开始日期
        end_date: 随机时间的结束日期
    """
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 不存在")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        sys.exit(1)
    
    # 检查数据格式
    if not isinstance(data, list):
        print("错误: JSON数据必须是一个数组")
        sys.exit(1)
    
    # 统计信息
    total_count = len(data)
    added_count = 0
    reset_count = 0
    skipped_count = 0
    
    # 为每条记录添加或重置时间
    for item in data:
        if not isinstance(item, dict):
            print(f"警告: 跳过非字典类型的数据项")
            continue
        
        # 检查是否已有时间字段
        if 'time' in item:
            if force_reset:
                item['time'] = generate_random_time(start_date, end_date)
                reset_count += 1
            else:
                skipped_count += 1
        else:
            item['time'] = generate_random_time(start_date, end_date)
            added_count += 1
    
    # 确定输出文件路径
    if output_file is None:
        output_file = input_file
    
    # 写入JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误: 写入文件失败 - {e}")
        sys.exit(1)
    
    # 打印统计信息
    print(f"\n处理完成!")
    print(f"文件: {input_file}")
    print(f"总记录数: {total_count}")
    print(f"新增时间: {added_count}")
    print(f"重置时间: {reset_count}")
    print(f"跳过记录: {skipped_count}")
    if output_file != input_file:
        print(f"输出文件: {output_file}")
    else:
        print(f"已更新原文件")
    print()


def process_directory(directory, pattern="*.json", force_reset=False,
                      start_date="2025-09-01", end_date="2025-10-31"):
    """
    批量处理目录下的所有JSON文件
    
    Args:
        directory: 目录路径
        pattern: 文件匹配模式
        force_reset: 是否强制重置已有的时间字段
        start_date: 随机时间的开始日期
        end_date: 随机时间的结束日期
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"错误: 目录 '{directory}' 不存在")
        sys.exit(1)
    
    json_files = list(dir_path.glob(pattern))
    
    if not json_files:
        print(f"警告: 在目录 '{directory}' 中未找到匹配 '{pattern}' 的文件")
        return
    
    print(f"\n找到 {len(json_files)} 个JSON文件")
    print("=" * 50)
    
    for json_file in json_files:
        add_time_to_json(
            str(json_file),
            force_reset=force_reset,
            start_date=start_date,
            end_date=end_date
        )


def main():
    parser = argparse.ArgumentParser(
        description='为JSON数据添加或重置随机时间字段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 为单个文件添加时间（不覆盖已有时间）
  python add_random_time.py -f data.json
  
  # 强制重置所有时间字段
  python add_random_time.py -f data.json --reset
  
  # 指定输出文件
  python add_random_time.py -f data.json -o output.json
  
  # 批量处理目录下所有JSON文件
  python add_random_time.py -d ./data --reset
  
  # 自定义时间范围
  python add_random_time.py -f data.json --start 2025-01-01 --end 2025-12-31
        '''
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', help='输入JSON文件路径')
    group.add_argument('-d', '--directory', help='批量处理目录路径')
    
    parser.add_argument('-o', '--output', help='输出文件路径（仅对单文件有效）')
    parser.add_argument('--reset', action='store_true', 
                       help='强制重置已有的时间字段')
    parser.add_argument('--pattern', default='*.json',
                       help='文件匹配模式（仅对目录有效，默认: *.json）')
    parser.add_argument('--start', default='2025-09-01',
                       help='随机时间开始日期 (默认: 2025-01-01)')
    parser.add_argument('--end', default='2025-10-31',
                       help='随机时间结束日期 (默认: 2025-10-31)')
    
    args = parser.parse_args()
    
    # 验证日期格式
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("错误: 日期格式必须为 YYYY-MM-DD")
        sys.exit(1)
    
    # 处理文件或目录
    if args.file:
        add_time_to_json(
            args.file,
            args.output,
            args.reset,
            args.start,
            args.end
        )
    else:
        if args.output:
            print("警告: --output 参数仅在处理单个文件时有效")
        process_directory(
            args.directory,
            args.pattern,
            args.reset,
            args.start,
            args.end
        )


if __name__ == '__main__':
    main()

