#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调整训练数据：将第一步function_call中的query改为直接使用human的value
"""
import json
import re

def process_conversations(data):
    """处理对话数据"""
    processed_count = 0
    
    for item in data:
        conversations = item.get("conversations", [])
        
        # 找到第一个human的value
        human_value = None
        for conv in conversations:
            if conv.get("from") == "human":
                human_value = conv.get("value", "")
                break
        
        if not human_value:
            continue
        
        # 找到第一个function_call并修改其query
        for conv in conversations:
            if conv.get("from") == "function_call":
                value = conv.get("value", "")
                
                # 检查是否是retrieval_tool的调用格式（包含query和source_filter）
                if '"query":' in value and '"source_filter":' in value:
                    # 提取source_filter的值
                    source_filter_match = re.search(r'"source_filter":\s*"([^"]+)"', value)
                    if source_filter_match:
                        source_filter = source_filter_match.group(1)
                        # 构建新的value，使用human的原始问题作为query
                        new_value = f'"query": "{human_value}", "source_filter": "{source_filter}"'
                        conv["value"] = new_value
                        processed_count += 1
                
                # 只处理第一个function_call
                break
    
    return processed_count

def main():
    input_file = "/home/ziqiang/LLaMA-Factory/data/dataset/11_07/11.05_train_data_processed.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/dataset/11_07/11.05_train_data_processed_updated.json"
    
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据条数: {len(data)}")
    
    # 处理数据
    processed_count = process_conversations(data)
    print(f"已处理 {processed_count} 条对话的第一步function_call")
    
    # 保存结果
    print(f"保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("处理完成！")
    
    # 显示第一个样本的前几行以供验证
    if data:
        print("\n第一个样本的前两个对话轮次:")
        for i, conv in enumerate(data[0]["conversations"][:3]):
            print(f"\n[{i}] from: {conv['from']}")
            value = conv.get('value', '')
            if len(value) > 200:
                print(f"value: {value[:200]}...")
            else:
                print(f"value: {value}")

if __name__ == "__main__":
    main()

