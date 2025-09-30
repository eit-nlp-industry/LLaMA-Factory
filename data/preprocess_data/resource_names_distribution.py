#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter

def analyze_resource_names_distribution(file_path):
    """专门分析resource_names字段的分布情况，包括单个资源和资源组合"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据条数: {len(data)}")
    print("=" * 60)
    
    # 统计resource_names字段的分布
    resource_names_counter = Counter()  # 单个资源的频次
    resource_combinations_counter = Counter()  # 资源组合的频次
    total_resource_names_entries = 0
    empty_resource_names = 0
    
    for item in data:
        output = item.get('output', '')
        
        try:
            output_json = json.loads(output)
            
            # 检查是否有resource_names字段
            if 'resource_names' in output_json:
                total_resource_names_entries += 1
                resource_names_list = output_json['resource_names']
                
                if not resource_names_list:  # 空列表
                    empty_resource_names += 1
                else:
                    # 统计每个资源名称
                    for resource in resource_names_list:
                        resource_names_counter[resource] += 1
                    
                    # 统计资源组合（将列表转换为排序的元组作为key）
                    if len(resource_names_list) > 1:
                        # 对资源名称排序以确保组合的一致性
                        sorted_combination = tuple(sorted(resource_names_list))
                        resource_combinations_counter[sorted_combination] += 1
                    elif len(resource_names_list) == 1:
                        # 单个资源也算作一种"组合"
                        single_resource = tuple(resource_names_list)
                        resource_combinations_counter[single_resource] += 1
                        
        except json.JSONDecodeError:
            continue
    
    print(f"🎯 resource_names字段分布统计:")
    print(f"包含resource_names字段的条目数: {total_resource_names_entries}")
    print(f"resource_names为空列表的条目数: {empty_resource_names}")
    print(f"不同资源名称的种类数: {len(resource_names_counter)}")
    print(f"不同资源组合的种类数: {len(resource_combinations_counter)}")
    print("=" * 60)
    
    # 1. 单个资源名称频次统计
    print("📊 各资源名称出现频次（按频次降序）:")
    print("-" * 60)
    sorted_resources = sorted(resource_names_counter.items(), key=lambda x: x[1], reverse=True)
    
    for i, (resource, count) in enumerate(sorted_resources, 1):
        print(f"{i:3d}. {resource}: {count}")
    
    # 2. 资源组合频次统计
    print(f"\n🔗 资源组合出现频次（按频次降序）:")
    print("-" * 60)
    sorted_combinations = sorted(resource_combinations_counter.items(), key=lambda x: x[1], reverse=True)
    
    for i, (combination, count) in enumerate(sorted_combinations, 1):
        if len(combination) == 1:
            combo_str = f"[单个资源] {combination[0]}"
        else:
            combo_str = f"[{len(combination)}个资源] " + " + ".join(combination)
        print(f"{i:3d}. {combo_str}: {count}")
    
    # 按资源主体分类统计
    print(f"\n📋 按资源主体分类统计:")
    print("-" * 60)
    
    resource_by_category = {}
    for resource, count in resource_names_counter.items():
        if '-' in resource:
            category = resource.split('-')[0]
            if category not in resource_by_category:
                resource_by_category[category] = {}
            resource_by_category[category][resource] = count
        else:
            # 没有'-'的资源名称
            if '其他' not in resource_by_category:
                resource_by_category['其他'] = {}
            resource_by_category['其他'][resource] = count
    
    # 按类别输出
    for category, resources in sorted(resource_by_category.items()):
        total_in_category = sum(resources.values())
        print(f"\n{category} (总计: {total_in_category}):")
        for resource, count in sorted(resources.items(), key=lambda x: x[1], reverse=True):
            print(f"  {resource}: {count}")
    
    return resource_names_counter, resource_combinations_counter, total_resource_names_entries, empty_resource_names

if __name__ == "__main__":
    file_path = "/home/ziqiang/LLaMA-Factory/data/ocr_text_orders_08_14_test_v4.json"
    analyze_resource_names_distribution(file_path)
