#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去除 function_call_context_audit.json 中的重复数据
"""

import json
from typing import Dict, Any, List


def get_item_key(item: Dict[str, Any]) -> str:
    """
    生成数据项的唯一键
    使用 query + function_call 的组合作为键
    """
    if 'conversations' not in item or len(item['conversations']) < 2:
        return ""
    
    query = item['conversations'][0].get('value', '')
    function_call = item['conversations'][1].get('value', '') if len(item['conversations']) > 1 else ''
    
    return f"{query}||{function_call}"


def deduplicate_data(data: List[Dict[str, Any]], keep_strategy: str = 'first') -> List[Dict[str, Any]]:
    """
    去重数据
    
    Args:
        data: 原始数据列表
        keep_strategy: 保留策略
            - 'first': 保留第一次出现的
            - 'last': 保留最后一次出现的
    
    Returns:
        去重后的数据列表
    """
    seen = {}
    
    for i, item in enumerate(data):
        key = get_item_key(item)
        if not key:
            continue
        
        if key not in seen:
            seen[key] = i
        elif keep_strategy == 'last':
            seen[key] = i
    
    # 保持原始顺序
    result = []
    if keep_strategy == 'first':
        for i, item in enumerate(data):
            key = get_item_key(item)
            if key and seen.get(key) == i:
                result.append(item)
    else:  # last
        indices_to_keep = set(seen.values())
        for i, item in enumerate(data):
            if i in indices_to_keep:
                result.append(item)
    
    return result


def main():
    """主函数"""
    input_file = "/home/ziqiang/LLaMA-Factory/data/function_call_data/function_call_context_audit.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/function_call_data/function_call_context_audit_deduplicated.json"
    backup_file = "/home/ziqiang/LLaMA-Factory/data/function_call_data/function_call_context_audit.json.backup"
    
    print("读取数据...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据: {len(data)} 条")
    
    # 统计重复情况
    queries = []
    for item in data:
        if 'conversations' in item and len(item['conversations']) > 0:
            query = item['conversations'][0].get('value', '')
            queries.append(query)
    
    from collections import Counter
    query_counts = Counter(queries)
    duplicates = {q: count for q, count in query_counts.items() if count > 1}
    
    print(f"唯一query: {len(set(queries))} 个")
    print(f"重复query: {len(duplicates)} 个")
    print(f"重复条数: {len(queries) - len(set(queries))} 条")
    
    # 去重
    print("\n执行去重...")
    deduped_data = deduplicate_data(data, keep_strategy='first')
    
    print(f"去重后数据: {len(deduped_data)} 条")
    print(f"删除了: {len(data) - len(deduped_data)} 条")
    
    # 备份原文件
    print(f"\n备份原文件到: {backup_file}")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 保存去重后的数据
    print(f"保存去重后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduped_data, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 完成！")
    
    # 验证
    with open(output_file, 'r', encoding='utf-8') as f:
        verified_data = json.load(f)
    
    verified_queries = []
    for item in verified_data:
        if 'conversations' in item and len(item['conversations']) > 0:
            query = item['conversations'][0].get('value', '')
            verified_queries.append(query)
    
    verified_counts = Counter(verified_queries)
    verified_duplicates = {q: count for q, count in verified_counts.items() if count > 1}
    
    print(f"\n验证结果:")
    print(f"  总条数: {len(verified_data)}")
    print(f"  唯一query: {len(set(verified_queries))}")
    print(f"  仍有重复: {len(verified_duplicates)} 个")
    
    if len(verified_duplicates) > 0:
        print("\n  注意：以下query仍有重复（可能是function_call不同）:")
        for q, count in list(verified_duplicates.items())[:5]:
            print(f"    - \"{q}\" - {count}次")


if __name__ == "__main__":
    main()

