#!/usr/bin/env python3
"""
修复训练数据格式，将function_call格式转换为assistant格式
"""

import json
import os

def fix_data_format(input_file, output_file):
    """修复数据格式"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_data = []
    
    for item in data:
        conversations = item.get('conversations', [])
        if not conversations:
            continue
            
        fixed_conversations = []
        
        for i, conv in enumerate(conversations):
            if conv['from'] == 'function_call':
                # 将function_call转换为assistant格式
                fixed_conversations.append({
                    'from': 'assistant',
                    'value': conv['value']
                })
            else:
                fixed_conversations.append(conv)
        
        fixed_item = {
            'conversations': fixed_conversations,
            'system': item.get('system', ''),
            'tools': item.get('tools', '')
        }
        
        fixed_data.append(fixed_item)
    
    # 保存修复后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据格式修复完成")
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 修复样本数: {len(fixed_data)}")

if __name__ == "__main__":
    input_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo_step1.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/dataset/9_17/demo_step1_fixed.json"
    
    fix_data_format(input_file, output_file)





