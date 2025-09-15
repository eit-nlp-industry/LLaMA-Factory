#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可配置的JSON数据集分割脚本
功能：将JSON数据集按指定比例分割为训练集和测试集
支持命令行参数和配置文件
"""

import json
import random
import os
import argparse
from pathlib import Path

def split_dataset(data, train_ratio=0.7, random_seed=42):
    """
    分割数据集
    
    Args:
        data (list): 输入数据列表
        train_ratio (float): 训练集比例，默认0.7
        random_seed (int): 随机种子，确保结果可重现
    
    Returns:
        tuple: (训练集数据, 测试集数据)
    """
    # 设置随机种子
    random.seed(random_seed)
    
    print(f"总数据量: {len(data)} 条")
    
    # 打乱数据
    print("正在打乱数据...")
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算分割点
    split_point = int(len(shuffled_data) * train_ratio)
    
    # 分割数据
    train_data = shuffled_data[:split_point]
    test_data = shuffled_data[split_point:]
    
    print(f"训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.1f}%)")
    print(f"测试集: {len(test_data)} 条 ({len(test_data)/len(data)*100:.1f}%)")
    
    return train_data, test_data

def save_json(data, output_file, ensure_ascii=False):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        output_file (str): 输出文件路径
        ensure_ascii (bool): 是否确保ASCII编码
    """
    print(f"正在保存到: {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=2)
    print(f"保存完成: {output_file}")

def load_json(input_file):
    """
    读取JSON文件
    
    Args:
        input_file (str): 输入文件路径
    
    Returns:
        list: JSON数据
    """
    print(f"正在读取文件: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON文件应该包含一个数组")
    
    return data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='JSON数据集分割工具')
    parser.add_argument('input_file', help='输入JSON文件路径')
    parser.add_argument('-t', '--train-ratio', type=float, default=0.7, 
                       help='训练集比例 (默认: 0.7)')
    parser.add_argument('-s', '--seed', type=int, default=42, 
                       help='随机种子 (默认: 42)')
    parser.add_argument('--train-output', default='answers_train.json', 
                       help='训练集输出文件名 (默认: answers_train.json)')
    parser.add_argument('--test-output', default='answers_test.json', 
                       help='测试集输出文件名 (默认: answers_test.json)')
    parser.add_argument('--ensure-ascii', action='store_true', 
                       help='确保输出为ASCII编码')
    parser.add_argument('--output-dir', default='', 
                       help='输出目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    try:
        # 读取数据
        data = load_json(args.input_file)
        
        # 检查训练集比例
        if not 0 < args.train_ratio < 1:
            raise ValueError("训练集比例必须在0和1之间")
        
        # 分割数据集
        train_data, test_data = split_dataset(
            data, 
            train_ratio=args.train_ratio, 
            random_seed=args.seed
        )
        
        # 构建输出文件路径
        if args.output_dir:
            train_file = os.path.join(args.output_dir, args.train_output)
            test_file = os.path.join(args.output_dir, args.test_output)
        else:
            train_file = args.train_output
            test_file = args.test_output
        
        # 保存训练集
        save_json(train_data, train_file, args.ensure_ascii)
        
        # 保存测试集
        save_json(test_data, test_file, args.ensure_ascii)
        
        print("\n数据集分割完成！")
        print(f"训练集文件: {train_file}")
        print(f"测试集文件: {test_file}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"- 原始数据: {len(data)} 条")
        print(f"- 训练集: {len(train_data)} 条")
        print(f"- 测试集: {len(test_data)} 条")
        print(f"- 分割比例: {args.train_ratio:.1f}:{1-args.train_ratio:.1f}")
        print(f"- 随机种子: {args.seed}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
