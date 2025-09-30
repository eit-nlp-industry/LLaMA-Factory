#!/usr/bin/env python3
"""
分析Token调试日志的脚本
用于分析训练日志中的token处理情况
"""

import re
import sys
from collections import defaultdict

def analyze_token_logs(log_file):
    """分析token调试日志"""
    print(f"=== 分析日志文件: {log_file} ===\n")
    
    # 统计信息
    stats = {
        'total_samples': 0,
        'truncated_pairs': 0,
        'dropped_pairs': 0,
        'cutoff_usage': [],
        'truncation_events': []
    }
    
    current_sample = None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测新样本开始
            if '[TOKEN_DEBUG] 开始处理数据样本' in line:
                stats['total_samples'] += 1
                current_sample = {
                    'sample_id': stats['total_samples'],
                    'pairs': [],
                    'final_length': 0,
                    'cutoff_len': 0
                }
            
            # 记录cutoff_len
            elif '[TOKEN_DEBUG] cutoff_len:' in line:
                cutoff_match = re.search(r'cutoff_len: (\d+)', line)
                if cutoff_match:
                    current_sample['cutoff_len'] = int(cutoff_match.group(1))
            
            # 记录pair信息
            elif '[TOKEN_DEBUG] === Pair' in line:
                pair_match = re.search(r'Pair (\d+)', line)
                if pair_match:
                    pair_id = int(pair_match.group(1))
                    current_sample['pairs'].append({
                        'id': pair_id,
                        'original_source': 0,
                        'original_target': 0,
                        'truncated_source': 0,
                        'truncated_target': 0,
                        'truncated': False
                    })
            
            # 记录原始长度
            elif '[TOKEN_DEBUG] 原始长度:' in line:
                length_match = re.search(r'source=(\d+), target=(\d+)', line)
                if length_match and current_sample and current_sample['pairs']:
                    source_len = int(length_match.group(1))
                    target_len = int(length_match.group(2))
                    current_sample['pairs'][-1]['original_source'] = source_len
                    current_sample['pairs'][-1]['original_target'] = target_len
            
            # 记录截断后长度
            elif '[TOKEN_DEBUG] 截断后长度:' in line:
                truncate_match = re.search(r'source=(\d+)->(\d+), target=(\d+)->(\d+)', line)
                if truncate_match and current_sample and current_sample['pairs']:
                    orig_source = int(truncate_match.group(1))
                    new_source = int(truncate_match.group(2))
                    orig_target = int(truncate_match.group(3))
                    new_target = int(truncate_match.group(4))
                    
                    current_sample['pairs'][-1]['truncated_source'] = new_source
                    current_sample['pairs'][-1]['truncated_target'] = new_target
                    
                    if new_source < orig_source or new_target < orig_target:
                        current_sample['pairs'][-1]['truncated'] = True
                        stats['truncated_pairs'] += 1
                        stats['truncation_events'].append({
                            'sample_id': current_sample['sample_id'],
                            'pair_id': current_sample['pairs'][-1]['id'],
                            'source_truncated': new_source < orig_source,
                            'target_truncated': new_target < orig_target
                        })
            
            # 记录预算耗尽
            elif '[TOKEN_DEBUG] 预算耗尽，丢弃剩余pairs' in line:
                stats['dropped_pairs'] += 1
            
            # 记录最终结果
            elif '[TOKEN_DEBUG] 最终total_length:' in line:
                final_match = re.search(r'最终total_length: (\d+)', line)
                if final_match and current_sample:
                    current_sample['final_length'] = int(final_match.group(1))
                    stats['cutoff_usage'].append({
                        'sample_id': current_sample['sample_id'],
                        'used': current_sample['final_length'],
                        'cutoff': current_sample['cutoff_len'],
                        'usage_rate': current_sample['final_length'] / current_sample['cutoff_len'] if current_sample['cutoff_len'] > 0 else 0
                    })
    
    # 输出分析结果
    print(f"总样本数: {stats['total_samples']}")
    print(f"发生截断的pairs: {stats['truncated_pairs']}")
    print(f"预算耗尽的样本: {stats['dropped_pairs']}")
    print()
    
    if stats['cutoff_usage']:
        usage_rates = [item['usage_rate'] for item in stats['cutoff_usage']]
        avg_usage = sum(usage_rates) / len(usage_rates)
        max_usage = max(usage_rates)
        min_usage = min(usage_rates)
        
        print(f"Token使用率统计:")
        print(f"  平均使用率: {avg_usage:.1%}")
        print(f"  最大使用率: {max_usage:.1%}")
        print(f"  最小使用率: {min_usage:.1%}")
        print()
    
    # 截断事件分析
    if stats['truncation_events']:
        print("截断事件分析:")
        source_truncated = sum(1 for event in stats['truncation_events'] if event['source_truncated'])
        target_truncated = sum(1 for event in stats['truncation_events'] if event['target_truncated'])
        
        print(f"  source被截断: {source_truncated} 次")
        print(f"  target被截断: {target_truncated} 次")
        print()
        
        # 显示前几个截断事件
        print("前5个截断事件:")
        for i, event in enumerate(stats['truncation_events'][:5]):
            print(f"  样本{event['sample_id']} Pair{event['pair_id']}: "
                  f"source截断={event['source_truncated']}, target截断={event['target_truncated']}")
    
    # 使用率分布
    if stats['cutoff_usage']:
        print("\n使用率分布:")
        ranges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0), (1.0, 1.0)]
        for start, end in ranges:
            count = sum(1 for item in stats['cutoff_usage'] 
                       if start <= item['usage_rate'] < end or (end == 1.0 and item['usage_rate'] == 1.0))
            print(f"  {start:.0%}-{end:.0%}: {count} 个样本")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python analyze_token_logs.py <log_file>")
        print("示例: python analyze_token_logs.py debug_train.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    try:
        analyze_token_logs(log_file)
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_file}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
