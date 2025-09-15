#!/usr/bin/env python3
"""
预处理脚本：为训练数据添加时间上下文信息
"""

import json
import re
from typing import Dict, List, Any


def extract_time_from_input(input_text: str) -> str:
    """从input文本中提取时间信息"""
    time_str = "未知时间"
    
    # 日期模式匹配
    date_patterns = [
        r'(\d{1,2}月\d{1,2}日)',  # 7月15日
        r'(\d{4}-\d{1,2}-\d{1,2})',  # 2025-07-15  
        r'(\d{1,2}\.\d{1,2})',  # 7.15
        r'日期(\d{1,2}\.\d{1,2})',  # 日期7.16
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, input_text)
        if match:
            time_str = match.group(1)
            # 如果是数字格式，转换为更友好的格式
            if re.match(r'\d{1,2}\.\d{1,2}', time_str):
                month, day = time_str.split('.')
                time_str = f"7月{day}日"  # 假设都是7月
            break
    
    return time_str


def find_end_date_for_input(input_text: str, all_examples: List[Dict[str, Any]]) -> str:
    """为给定的input查找对应的end_date"""
    for example in all_examples:
        if (example.get('input') == input_text and 
            'end_date' in example.get('instruction', '') and
            example.get('output')):
            try:
                output_data = json.loads(example['output'])
                if 'end_date' in output_data and output_data['end_date']:
                    date_str = output_data['end_date']
                    # 从 "2025-07-17 00:00:00" 提取 "7月17日"
                    date_match = re.search(r'2025-(\d{2})-(\d{2})', date_str)
                    if date_match:
                        month = int(date_match.group(1))
                        day = int(date_match.group(2))
                        return f"{month}月{day}日"
            except (json.JSONDecodeError, KeyError):
                continue
    return "未知时间"


def create_context_enhanced_data(input_file: str, output_file: str):
    """创建包含时间上下文的训练数据"""
    print(f"Reading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # 创建input到时间的映射
    input_to_time = {}
    
    print("Building time mapping from end_date examples...")
    for example in data:
        input_text = example.get('input', '')
        if 'end_date' in example.get('instruction', ''):
            time_str = find_end_date_for_input(input_text, [example])
            if time_str != "未知时间":
                input_to_time[input_text] = time_str
    
    print(f"Found time information for {len(input_to_time)} unique inputs")
    
    # 处理每个样本
    enhanced_data = []
    for i, example in enumerate(data):
        if i % 100 == 0:
            print(f"Processing example {i+1}/{len(data)}")
        
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        # 获取时间信息
        time_str = input_to_time.get(input_text)
        if not time_str:
            time_str = extract_time_from_input(input_text)
        
        # 构建context
        context = f'context: {{"time": "{time_str}"}}'
        
        # 构建三段式prompt
        new_instruction = f"{context}\n\ninput: {input_text}\n\ninstruction: {instruction}"
        
        enhanced_example = {
            "instruction": new_instruction,
            "input": "",  # 将input内容移到instruction中
            "output": output
        }
        
        enhanced_data.append(enhanced_example)
    
    print(f"Writing enhanced data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"Done! Created {len(enhanced_data)} enhanced examples")
    
    # 打印几个示例
    print("\n=== Sample Enhanced Examples ===")
    for i in range(min(3, len(enhanced_data))):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {enhanced_data[i]['instruction'][:200]}...")
        print(f"Output: {enhanced_data[i]['output'][:100]}...")


if __name__ == "__main__":
    input_file = "ocr_text_orders_fine_grained_test_08_07_updated_v2_filter.json"
    output_file = "ocr_text_orders_fine_grained_test_08_07_updated_v2_filter_enhanced.json"
    
    create_context_enhanced_data(input_file, output_file)
