#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据集分割脚本
功能：将JSON数据集按7:3比例分割为训练集和测试集
"""

import json
import random
import os
from pathlib import Path

def split_dataset(input_file, train_ratio=0.7, random_seed=42):
    """
    分割JSON数据集
    
    Args:
        input_file (str): 输入JSON文件路径
        train_ratio (float): 训练集比例，默认0.7
        random_seed (int): 随机种子，确保结果可重现
    
    Returns:
        tuple: (训练集数据, 测试集数据)
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 读取JSON文件
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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

def save_json(data, output_file):
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        output_file (str): 输出文件路径
    """
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {output_file}")

def main():
    """主函数"""
    # 输入文件路径
    input_file = "price_policy_step1.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        return
    
    # 输出文件路径
    train_file = "train_dataset.json"
    test_file = "test_dataset.json"
    
    try:
        # 分割数据集
        train_data, test_data = split_dataset(input_file, train_ratio=0.7, random_seed=42)
        
        # 保存训练集
        save_json(train_data, train_file)
        
        # 保存测试集
        save_json(test_data, test_file)
        
        print("\n数据集分割完成！")
        print(f"训练集文件: {train_file}")
        print(f"测试集文件: {test_file}")
        
        # 显示一些统计信息
        print(f"\n统计信息:")
        print(f"- 原始数据: {len(train_data) + len(test_data)} 条")
        print(f"- 训练集: {len(train_data)} 条")
        print(f"- 测试集: {len(test_data)} 条")
        print(f"- 分割比例: 7:3")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()
