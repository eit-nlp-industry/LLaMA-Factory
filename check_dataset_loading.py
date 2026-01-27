#!/usr/bin/env python3
"""
检查LLaMA-Factory实际加载了多少数据
运行训练前执行此脚本可以预先知道哪些数据会被过滤
"""

import json
import sys
sys.path.insert(0, '/home/ziqiang/LLaMA-Factory')

from datasets import load_dataset

print("🔍 模拟LLaMA-Factory数据加载过程\n")

# 加载数据集配置
dataset_info_path = 'data/dataset_info.json'
with open(dataset_info_path, 'r', encoding='utf-8') as f:
    dataset_info = json.load(f)

dataset_name = 'data_demo'
if dataset_name in dataset_info:
    info = dataset_info[dataset_name]
    file_path = info['file_name']
    
    print(f"📂 数据集: {dataset_name}")
    print(f"📄 文件路径: {file_path}")
    print()
    
    # 读取原始数据
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"📊 原始数据: {len(raw_data)} 条")
    
    # 检查数据问题
    valid_data = []
    filtered_reasons = {
        'empty_conversation': [],
        'invalid_format': [],
        'duplicate': [],
        'empty_message': []
    }
    
    seen_conversations = set()
    
    for idx, item in enumerate(raw_data):
        is_valid = True
        
        # 检查conversations字段
        if 'conversations' not in item:
            filtered_reasons['invalid_format'].append(idx)
            is_valid = False
            continue
        
        conversations = item['conversations']
        
        # 检查是否为空
        if not conversations or len(conversations) == 0:
            filtered_reasons['empty_conversation'].append(idx)
            is_valid = False
            continue
        
        # 检查对话内容
        has_empty = False
        for conv in conversations:
            if not isinstance(conv, dict):
                filtered_reasons['invalid_format'].append(idx)
                is_valid = False
                break
            
            if 'value' not in conv or not conv['value'] or conv['value'].strip() == '':
                has_empty = True
                break
        
        if has_empty:
            filtered_reasons['empty_message'].append(idx)
            is_valid = False
            continue
        
        # 检查重复
        conv_str = json.dumps(conversations, sort_keys=True, ensure_ascii=False)
        if conv_str in seen_conversations:
            filtered_reasons['duplicate'].append(idx)
            is_valid = False
            continue
        
        seen_conversations.add(conv_str)
        
        if is_valid:
            valid_data.append(item)
    
    print(f"✅ 有效数据: {len(valid_data)} 条")
    print(f"❌ 过滤数据: {len(raw_data) - len(valid_data)} 条")
    print()
    
    # 显示过滤原因
    total_filtered = 0
    for reason, indices in filtered_reasons.items():
        if indices:
            total_filtered += len(indices)
            print(f"  - {reason}: {len(indices)} 条")
            if len(indices) <= 5:
                print(f"    索引: {indices}")
            else:
                print(f"    索引: {indices[:5]} ... (还有{len(indices)-5}条)")
    
    print()
    print("=" * 60)
    print("📊 训练步数预测:")
    print("=" * 60)
    
    # 计算训练步数
    num_gpus = 2
    per_device_batch = 1
    grad_accum = 8
    epochs = 10
    
    # DDP分配
    samples_per_gpu = len(valid_data) // num_gpus
    steps_per_epoch = samples_per_gpu // (per_device_batch * grad_accum)
    total_steps = steps_per_epoch * epochs
    
    print(f"有效数据: {len(valid_data)} 条")
    print(f"每GPU数据: {samples_per_gpu} 条")
    print(f"每epoch步数: {steps_per_epoch}")
    print(f"总步数 ({epochs} epochs): {total_steps}")
    print()
    
    # 对比
    print("💡 如果实际显示1000步:")
    expected_steps = 1000 / epochs
    expected_data = int(expected_steps * per_device_batch * grad_accum * num_gpus)
    print(f"   期望每epoch: {expected_steps:.0f} 步")
    print(f"   期望使用数据: {expected_data} 条")
    print(f"   差异: {len(valid_data) - expected_data} 条")
    
else:
    print(f"❌ 未找到数据集: {dataset_name}")

