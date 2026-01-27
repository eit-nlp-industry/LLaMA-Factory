#!/usr/bin/env python3
"""
数据集一致性检查脚本
检查训练数据集的token数量和样本数量
"""

import json
import os
from pathlib import Path
from transformers import AutoTokenizer

def check_dataset_consistency():
    """检查数据集一致性"""
    
    print("=" * 60)
    print("🔍 数据集一致性检查")
    print("=" * 60)
    
    # 数据集路径
    data_dir = Path("/home/ziqiang/LLaMA-Factory/data/data_demo")
    
    if not data_dir.exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        return
    
    # 统计JSON文件
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 未找到JSON文件: {data_dir}")
        return
    
    print(f"\n📁 数据集目录: {data_dir}")
    print(f"📄 找到 {len(json_files)} 个JSON文件\n")
    
    # 加载tokenizer
    print("🔧 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/models/Qwen3-8B",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"⚠️ 无法加载tokenizer: {e}")
        print("   将只统计样本数量，不统计tokens\n")
        tokenizer = None
    
    total_samples = 0
    total_tokens = 0
    file_stats = []
    
    # 处理每个JSON文件
    for json_file in sorted(json_files):
        print(f"📄 处理: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同格式
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict) and 'data' in data:
                samples = data['data']
            else:
                samples = [data]
            
            file_samples = len(samples)
            total_samples += file_samples
            
            # 统计tokens
            file_tokens = 0
            if tokenizer:
                for sample in samples:
                    # 尝试提取文本内容
                    text = ""
                    if isinstance(sample, dict):
                        # 常见的字段名
                        for key in ['instruction', 'input', 'output', 'conversation', 'messages']:
                            if key in sample:
                                if isinstance(sample[key], str):
                                    text += sample[key] + " "
                                elif isinstance(sample[key], list):
                                    for item in sample[key]:
                                        if isinstance(item, dict) and 'content' in item:
                                            text += item['content'] + " "
                    elif isinstance(sample, str):
                        text = sample
                    
                    if text:
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        file_tokens += len(tokens)
            
            total_tokens += file_tokens
            file_stats.append({
                'file': json_file.name,
                'samples': file_samples,
                'tokens': file_tokens
            })
            
            print(f"   ✅ 样本数: {file_samples}, Tokens: {file_tokens:,}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    # 输出统计结果
    print("\n" + "=" * 60)
    print("📊 数据集统计汇总")
    print("=" * 60)
    print(f"总样本数: {total_samples:,}")
    if tokenizer:
        print(f"总Tokens: {total_tokens:,}")
        print(f"平均每样本Tokens: {total_tokens // total_samples if total_samples > 0 else 0:,}")
    
    # 与已知训练数据对比
    print("\n" + "=" * 60)
    print("🔍 与训练日志对比")
    print("=" * 60)
    
    known_stats = {
        "最佳单卡(10-29)": {"tokens": 52320, "step": 1},
        "4卡训练(11-08)": {"tokens": 49584, "step": 1},
        "单卡8epoch(11-07)": {"tokens": 49584, "step": 1},
    }
    
    if tokenizer:
        # 估算每步的tokens（基于batch_size=16）
        estimated_step1_tokens = total_tokens // (total_samples // 16) if total_samples >= 16 else total_tokens
        
        print(f"\n当前数据集估算:")
        print(f"  每步Tokens (batch=16): ~{estimated_step1_tokens:,}")
        
        print(f"\n历史训练数据:")
        for name, stats in known_stats.items():
            match = "✅" if abs(estimated_step1_tokens - stats['tokens']) < 1000 else "❌"
            print(f"  {match} {name}: {stats['tokens']:,} tokens (step {stats['step']})")
        
        # 判断数据集版本
        print(f"\n🎯 数据集版本判断:")
        if abs(estimated_step1_tokens - 52320) < 1000:
            print("  ✅ 数据集与最佳单卡训练一致（52,320 tokens版本）")
        elif abs(estimated_step1_tokens - 49584) < 1000:
            print("  ⚠️  数据集与4卡训练一致（49,584 tokens版本）")
            print("  💡 建议：使用52,320 tokens版本的数据集以获得最佳效果")
        else:
            print(f"  ⚠️  数据集大小异常: {estimated_step1_tokens:,} tokens")
            print("  💡 建议：检查数据集是否正确")
    
    # 保存统计结果
    stats_file = data_dir.parent / "dataset_stats.json"
    stats_data = {
        "total_samples": total_samples,
        "total_tokens": total_tokens if tokenizer else None,
        "files": file_stats,
        "estimated_step1_tokens": estimated_step1_tokens if tokenizer else None
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 统计结果已保存: {stats_file}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_dataset_consistency()















