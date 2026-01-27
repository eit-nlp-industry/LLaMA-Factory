#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level Loss Analysis Script with Context Window Analysis

增强版分析脚本，支持上下文窗口分析，可以识别：
- 特定token在什么上下文中loss高
- 上下文模式识别（如"anchor结构闭合位置"）
- 上下文loss分布分析
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def load_token_loss_data(token_loss_dir: str) -> List[Dict[str, Any]]:
    """Load all token loss records from JSONL files."""
    records = []
    token_loss_dir = Path(token_loss_dir)
    
    # Find all JSONL files
    jsonl_files = list(token_loss_dir.glob("token_losses_step_*.jsonl"))
    jsonl_files.sort()
    
    print(f"📂 Found {len(jsonl_files)} token loss data files")
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    
    print(f"✅ Loaded {len(records)} token loss records")
    return records


def analyze_context_patterns(records: List[Dict[str, Any]], top_k: int = 20):
    """分析token在特定上下文中的loss模式"""
    print("\n" + "="*80)
    print("📊 上下文模式分析")
    print("="*80)
    
    # 按token和上下文模式分组
    pattern_stats = defaultdict(lambda: {"losses": [], "count": 0, "contexts": []})
    
    for record in records:
        token = record.get("gt_token", "")
        token_id = record.get("gt_token_id", 0)
        loss = record.get("gt_token_loss", 0.0)
        
        # 构建上下文模式
        left_context = record.get("left_context_tokens", [])
        right_context = record.get("right_context_tokens", [])
        
        # 提取关键上下文特征
        # 1. 左侧最后一个token
        left_last = left_context[-1] if left_context else ""
        # 2. 右侧第一个token
        right_first = right_context[0] if right_context else ""
        # 3. 右侧前几个token的组合（用于识别结构模式）
        right_pattern = " ".join(right_context[:3]) if len(right_context) >= 3 else " ".join(right_context)
        
        # 创建模式key
        pattern_key = f"{token}|left:{left_last}|right:{right_first}|pattern:{right_pattern[:30]}"
        
        pattern_stats[pattern_key]["losses"].append(loss)
        pattern_stats[pattern_key]["count"] += 1
        pattern_stats[pattern_key]["contexts"].append({
            "left": left_context[-3:] if len(left_context) >= 3 else left_context,
            "right": right_context[:3] if len(right_context) >= 3 else right_context,
        })
    
    # 计算统计信息
    results = []
    for pattern_key, stats in pattern_stats.items():
        if stats["count"] >= 3:  # 至少出现3次
            results.append({
                "pattern": pattern_key,
                "count": stats["count"],
                "avg_loss": np.mean(stats["losses"]),
                "std_loss": np.std(stats["losses"]),
                "max_loss": np.max(stats["losses"]),
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values("avg_loss", ascending=False).head(top_k)
    
    print(f"\n🔝 Top {top_k} 高Loss上下文模式 (至少出现3次):")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"\n模式: {row['pattern']}")
        print(f"  出现次数: {row['count']}")
        print(f"  平均Loss: {row['avg_loss']:.4f} ± {row['std_loss']:.4f}")
        print(f"  最大Loss: {row['max_loss']:.4f}")
    
    return df


def analyze_context_loss_distribution(records: List[Dict[str, Any]], target_token: str = "}"):
    """分析特定token在不同上下文中的loss分布"""
    print("\n" + "="*80)
    print(f"📊 Token '{target_token}' 的上下文Loss分布分析")
    print("="*80)
    
    target_records = [r for r in records if r.get("gt_token") == target_token]
    
    if len(target_records) == 0:
        print(f"❌ 未找到token '{target_token}' 的记录")
        return None
    
    print(f"✅ 找到 {len(target_records)} 条记录")
    
    # 分析右侧上下文（通常结构闭合位置在右侧）
    right_context_patterns = defaultdict(lambda: {"losses": [], "count": 0})
    
    for record in target_records:
        right_context = record.get("right_context_tokens", [])
        right_losses = record.get("right_context_losses", [])
        
        # 提取右侧前3个token作为模式
        if len(right_context) >= 3:
            pattern = " ".join(right_context[:3])
            right_context_patterns[pattern]["losses"].append(record.get("gt_token_loss", 0.0))
            right_context_patterns[pattern]["count"] += 1
    
    # 统计
    results = []
    for pattern, stats in right_context_patterns.items():
        if stats["count"] >= 2:
            results.append({
                "right_context_pattern": pattern,
                "count": stats["count"],
                "avg_loss": np.mean(stats["losses"]),
                "std_loss": np.std(stats["losses"]),
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values("avg_loss", ascending=False)
    
    print(f"\n📈 右侧上下文模式统计 (至少出现2次):")
    print("-" * 80)
    print(df.to_string(index=False))
    
    return df


def visualize_context_loss(records: List[Dict[str, Any]], output_file: str = "context_loss_analysis.png"):
    """可视化上下文loss分布"""
    print("\n" + "="*80)
    print("📊 生成上下文Loss可视化")
    print("="*80)
    
    # 选择高loss的token记录
    high_loss_records = [r for r in records if r.get("gt_token_loss", 0) > 2.0]
    
    if len(high_loss_records) == 0:
        print("⚠️ 没有找到高loss记录")
        return
    
    # 分析左右上下文loss
    left_losses_all = []
    right_losses_all = []
    
    for record in high_loss_records[:100]:  # 限制前100条
        left_losses = record.get("left_context_losses", [])
        right_losses = record.get("right_context_losses", [])
        
        if left_losses:
            left_losses_all.extend(left_losses)
        if right_losses:
            right_losses_all.extend(right_losses)
    
    if left_losses_all or right_losses_all:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if left_losses_all:
            axes[0].hist(left_losses_all, bins=30, alpha=0.7, color='blue')
            axes[0].set_xlabel("Loss")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Left Context Loss Distribution")
            axes[0].grid(True, alpha=0.3)
        
        if right_losses_all:
            axes[1].hist(right_losses_all, bins=30, alpha=0.7, color='red')
            axes[1].set_xlabel("Loss")
            axes[1].set_ylabel("Frequency")
            axes[1].set_title("Right Context Loss Distribution")
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化图已保存: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze token-level loss data with context window analysis")
    parser.add_argument(
        "--token_loss_dir",
        type=str,
        required=True,
        help="Directory containing token loss JSONL files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--target_token",
        type=str,
        default="}",
        help="Target token to analyze in detail"
    )
    
    args = parser.parse_args()
    
    # Load data
    records = load_token_loss_data(args.token_loss_dir)
    
    if len(records) == 0:
        print("❌ No token loss records found!")
        return
    
    # Change to output directory
    os.chdir(args.output_dir)
    
    # Run analyses
    context_patterns_df = analyze_context_patterns(records, top_k=20)
    context_dist_df = analyze_context_loss_distribution(records, target_token=args.target_token)
    visualize_context_loss(records)
    
    # Save results
    if context_patterns_df is not None and len(context_patterns_df) > 0:
        context_patterns_df.to_csv("context_pattern_analysis.csv", index=False)
    
    if context_dist_df is not None and len(context_dist_df) > 0:
        context_dist_df.to_csv(f"context_distribution_{args.target_token}.csv", index=False)
    
    print("\n✅ 上下文分析完成！")
    print(f"📁 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
