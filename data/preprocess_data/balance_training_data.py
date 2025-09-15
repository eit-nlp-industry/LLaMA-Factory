#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练数据平衡脚本 - 针对旅游资源名称抽取任务

主要功能：
1. 分析当前数据分布
2. 对低频资源进行上采样
3. 生成数据增强样本
4. 输出平衡后的训练集
"""

import json
import random
import copy
from collections import Counter, defaultdict
from typing import List, Dict, Any
import argparse

class DataBalancer:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.data = self.load_data()
        self.resource_counts = Counter()
        self.resource_samples = defaultdict(list)
        self.analyze_distribution()
    
    def load_data(self) -> List[Dict]:
        """加载训练数据"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_distribution(self):
        """分析资源分布"""
        for idx, item in enumerate(self.data):
            if 'output' in item:
                try:
                    output_data = json.loads(item['output'])
                    if 'resource_names' in output_data:
                        resources = output_data['resource_names']
                        for resource in resources:
                            self.resource_counts[resource] += 1
                            self.resource_samples[resource].append(idx)
                except:
                    continue
    
    def get_balance_strategy(self, target_min_samples: int = 5) -> Dict[str, int]:
        """
        计算平衡策略
        
        Args:
            target_min_samples: 目标最小样本数
            
        Returns:
            Dict[资源名称, 需要增加的样本数]
        """
        balance_strategy = {}
        
        print("📊 当前资源分布分析:")
        print("-" * 50)
        
        for resource, count in self.resource_counts.most_common():
            if count < target_min_samples:
                needed = target_min_samples - count
                balance_strategy[resource] = needed
                print(f"❌ {resource}: {count} -> {target_min_samples} (需要+{needed})")
            else:
                print(f"✅ {resource}: {count}")
        
        return balance_strategy
    
    def create_augmented_sample(self, original_idx: int, target_resource: str) -> Dict[str, Any]:
        """
        创建数据增强样本
        
        策略：
        1. 保持原有的instruction不变
        2. 修改input中的关键信息（日期、人数、联系人等）
        3. 保持目标资源在output中
        """
        original = copy.deepcopy(self.data[original_idx])
        
        # 日期变换
        dates = ["7月15日", "7月16日", "7月19日", "7月21日", "7月22日", "7月25日", "7月26日", "8月1日", "8月2日", "8月5日"]
        
        # 人数变换
        people_counts = ["15人", "25人", "35人", "45人", "55人", "8人", "12人", "18人", "22人", "28人"]
        
        # 联系人变换（保持格式）
        phone_endings = ["1234", "5678", "9012", "3456", "7890", "2468", "1357", "9753", "8642", "0246"]
        
        input_text = original['input']
        
        # 简单的文本替换进行数据增强
        for date in ["7月17日", "7月18日", "7月20日", "7月28日", "7月29日", "7月30日", "7月31日"]:
            if date in input_text:
                input_text = input_text.replace(date, random.choice(dates))
                break
        
        # 替换人数
        import re
        people_pattern = r'\d+人'
        matches = re.findall(people_pattern, input_text)
        if matches:
            for match in matches:
                input_text = input_text.replace(match, random.choice(people_counts), 1)
        
        # 替换电话号码后四位
        phone_pattern = r'1[3-9]\d{9}'
        def replace_phone(match):
            phone = match.group()
            return phone[:-4] + random.choice(phone_endings)
        
        input_text = re.sub(phone_pattern, replace_phone, input_text)
        
        # 创建新样本
        new_sample = copy.deepcopy(original)
        new_sample['input'] = input_text
        
        return new_sample
    
    def balance_data(self, target_min_samples: int = 5) -> List[Dict[str, Any]]:
        """
        平衡数据集
        
        Args:
            target_min_samples: 目标最小样本数
            
        Returns:
            平衡后的数据集
        """
        balance_strategy = self.get_balance_strategy(target_min_samples)
        
        if not balance_strategy:
            print("✅ 数据已经平衡，无需调整")
            return self.data
        
        print(f"\n🔄 开始数据平衡，目标最小样本数: {target_min_samples}")
        print("-" * 50)
        
        balanced_data = copy.deepcopy(self.data)
        
        for resource, needed_count in balance_strategy.items():
            print(f"📈 正在增强 '{resource}' 的样本...")
            
            # 获取该资源的原始样本
            original_samples = self.resource_samples[resource]
            
            for i in range(needed_count):
                # 随机选择一个原始样本进行增强
                source_idx = random.choice(original_samples)
                augmented_sample = self.create_augmented_sample(source_idx, resource)
                balanced_data.append(augmented_sample)
            
            print(f"   ✅ 已添加 {needed_count} 个增强样本")
        
        return balanced_data
    
    def save_balanced_data(self, balanced_data: List[Dict], output_file: str):
        """保存平衡后的数据"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(balanced_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 已保存平衡后的数据到: {output_file}")
        print(f"   原始样本数: {len(self.data)}")
        print(f"   平衡后样本数: {len(balanced_data)}")
        print(f"   新增样本数: {len(balanced_data) - len(self.data)}")

def main():
    parser = argparse.ArgumentParser(description='平衡旅游资源训练数据')
    parser.add_argument('--input', required=True, help='输入的训练数据文件')
    parser.add_argument('--output', required=True, help='输出的平衡数据文件')
    parser.add_argument('--min-samples', type=int, default=5, help='目标最小样本数 (默认: 5)')
    
    args = parser.parse_args()
    
    print("🚀 开始数据平衡流程...")
    print("=" * 60)
    
    # 初始化平衡器
    balancer = DataBalancer(args.input)
    
    # 执行平衡
    balanced_data = balancer.balance_data(args.min_samples)
    
    # 保存结果
    balancer.save_balanced_data(balanced_data, args.output)
    
    print("\n🎉 数据平衡完成！")

if __name__ == "__main__":
    main()
