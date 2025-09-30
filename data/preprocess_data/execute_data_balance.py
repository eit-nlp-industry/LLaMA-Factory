#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行数据平衡的主脚本

结合你的具体需求：
- 新叶古村-新叶古村门票: 1 -> 5 (+4)
- 大慈岩-大慈岩索道: 2 -> 5 (+3) 
- 其他低频资源也会被相应增强
"""

import json
import sys
import os
from advanced_data_augmentation import AdvancedDataAugmenter

def load_training_data(file_path: str):
    """加载原始训练数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_enhanced_samples(original_data, enhanced_samples):
    """合并原始数据和增强样本"""
    return original_data + enhanced_samples

def analyze_final_distribution(data):
    """分析最终的数据分布"""
    from collections import Counter
    
    resource_counts = Counter()
    
    for item in data:
        if 'output' in item:
            try:
                output_data = json.loads(item['output'])
                if 'resource_names' in output_data:
                    resources = output_data['resource_names']
                    for resource in resources:
                        resource_counts[resource] += 1
            except:
                continue
    
    print("📊 最终数据分布:")
    print("-" * 60)
    
    # 重点关注的低频资源
    focus_resources = [
        "新叶古村-新叶古村门票",
        "大慈岩-大慈岩索道", 
        "灵栖洞-灵栖洞西游魔毯",
        "宿江公司-江清月近人实景演艺门票"
    ]
    
    print("🎯 重点关注的资源:")
    for resource in focus_resources:
        count = resource_counts.get(resource, 0)
        print(f"   {resource}: {count}")
    
    print(f"\n📈 所有资源分布 (总计 {len(resource_counts)} 种资源):")
    for resource, count in resource_counts.most_common():
        status = "✅" if count >= 5 else "⚠️"
        print(f"   {status} {resource}: {count}")

def main():
    # 输入文件路径 - 根据你的实际路径调整
    input_files = [
        "/home/ziqiang/LLaMA-Factory/data/ocr_text_orders_08_14_test_v4.json"
    ]
    
    # 尝试找到存在的训练数据文件
    training_file = None
    for file_path in input_files:
        if os.path.exists(file_path):
            training_file = file_path
            break
    
    if not training_file:
        print("❌ 未找到训练数据文件，请检查路径:")
        for file_path in input_files:
            print(f"   {file_path}")
        return
    
    print(f"📂 使用训练数据文件: {training_file}")
    print("=" * 60)
    
    # 加载原始数据
    print("📥 加载原始训练数据...")
    original_data = load_training_data(training_file)
    print(f"   原始样本数: {len(original_data)}")
    
    # 生成增强样本
    print("\n🔄 生成增强样本...")
    augmenter = AdvancedDataAugmenter()
    enhanced_samples = augmenter.generate_all_samples()
    
    # 合并数据
    print(f"\n🔗 合并原始数据和增强样本...")
    balanced_data = merge_enhanced_samples(original_data, enhanced_samples)
    print(f"   合并后样本数: {len(balanced_data)}")
    print(f"   新增样本数: {len(enhanced_samples)}")
    
    # 保存平衡后的数据
    output_file = "balanced_training_data.json"
    print(f"\n💾 保存平衡后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    
    # 分析最终分布
    print(f"\n📊 分析最终数据分布...")
    analyze_final_distribution(balanced_data)
    
    print(f"\n🎉 数据平衡完成！")
    print("📋 建议的下一步:")
    print("   1. 使用 balanced_training_data.json 重新训练模型")
    print("   2. 在验证集上测试性能改进")
    print("   3. 特别关注新叶古村、大慈岩索道等低频资源的识别效果")

if __name__ == "__main__":
    main()
