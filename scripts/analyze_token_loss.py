#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level Loss Analysis Script

This script analyzes token-level loss data collected during training to identify:
1. High-frequency, high-loss tokens
2. Token type clustering (structural, keyword, numeric, etc.)
3. Position-sensitive analysis
4. Top-k prediction comparison

Usage:
    python scripts/analyze_token_loss.py --token_loss_dir <path_to_token_loss_data>
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


def analyze_high_loss_tokens(records: List[Dict[str, Any]], top_k: int = 50) -> pd.DataFrame:
    """Analyze high-frequency, high-loss tokens."""
    print("\n" + "="*80)
    print("📊 第一步：高频高Loss Token统计")
    print("="*80)
    
    # Aggregate by token
    token_stats = defaultdict(lambda: {"losses": [], "count": 0, "token": "", "token_type": ""})
    
    for record in records:
        # Support both old and new field names
        token_id = record.get("gt_token_id") or record.get("token_id", 0)
        token_loss = record.get("gt_token_loss") or record.get("loss", 0.0)
        token_str = record.get("gt_token") or record.get("token", "")
        
        token_stats[token_id]["losses"].append(token_loss)
        token_stats[token_id]["count"] += 1
        token_stats[token_id]["token"] = token_str
        token_stats[token_id]["token_type"] = record.get("token_type", "unknown")
    
    # Compute statistics
    results = []
    for token_id, stats in token_stats.items():
        if stats["count"] > 0:
            results.append({
                "token_id": token_id,
                "token": stats["token"],
                "token_type": stats["token_type"],
                "count": stats["count"],
                "avg_loss": np.mean(stats["losses"]),
                "std_loss": np.std(stats["losses"]),
                "min_loss": np.min(stats["losses"]),
                "max_loss": np.max(stats["losses"]),
            })
    
    df = pd.DataFrame(results)
    
    # Sort by average loss (descending)
    df = df.sort_values("avg_loss", ascending=False)
    
    # Filter by frequency (at least 10 occurrences)
    df_filtered = df[df["count"] >= 10].head(top_k)
    
    print(f"\n🔝 Top {top_k} 高频高Loss Tokens (至少出现10次):")
    print("-" * 80)
    print(df_filtered[["token", "token_id", "token_type", "count", "avg_loss", "std_loss"]].to_string(index=False))
    
    return df_filtered


def analyze_token_type_clustering(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze loss by token type."""
    print("\n" + "="*80)
    print("📊 第二步：Token类型聚类分析")
    print("="*80)
    
    type_stats = defaultdict(lambda: {"losses": [], "count": 0})
    
    for record in records:
        token_type = record.get("token_type", "unknown")
        token_loss = record.get("gt_token_loss") or record.get("loss", 0.0)
        type_stats[token_type]["losses"].append(token_loss)
        type_stats[token_type]["count"] += 1
    
    results = []
    for token_type, stats in type_stats.items():
        if stats["count"] > 0:
            results.append({
                "token_type": token_type,
                "count": stats["count"],
                "avg_loss": np.mean(stats["losses"]),
                "std_loss": np.std(stats["losses"]),
                "min_loss": np.min(stats["losses"]),
                "max_loss": np.max(stats["losses"]),
            })
    
    df = pd.DataFrame(results)
    df = df.sort_values("avg_loss", ascending=False)
    
    print("\n📈 按Token类型统计的平均Loss:")
    print("-" * 80)
    print(df.to_string(index=False))
    
    return df


def analyze_position_sensitivity(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze loss by position in sequence."""
    print("\n" + "="*80)
    print("📊 第三步：位置敏感分析")
    print("="*80)
    
    position_stats = defaultdict(lambda: {"losses": [], "count": 0})
    
    for record in records:
        position = record.get("position", 0)
        token_loss = record.get("gt_token_loss") or record.get("loss", 0.0)
        position_stats[position]["losses"].append(token_loss)
        position_stats[position]["count"] += 1
    
    results = []
    for position, stats in sorted(position_stats.items()):
        if stats["count"] > 0:
            results.append({
                "position": position,
                "count": stats["count"],
                "avg_loss": np.mean(stats["losses"]),
                "std_loss": np.std(stats["losses"]),
            })
    
    df = pd.DataFrame(results)
    
    # Plot position vs loss
    if len(df) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(df["position"], df["avg_loss"], marker='o', markersize=3, linewidth=1)
        plt.xlabel("Position in Sequence")
        plt.ylabel("Average Loss")
        plt.title("Token Loss by Position in Sequence")
        plt.grid(True, alpha=0.3)
        
        # Highlight high-loss regions
        threshold = df["avg_loss"].quantile(0.9)
        high_loss_positions = df[df["avg_loss"] > threshold]
        if len(high_loss_positions) > 0:
            plt.scatter(high_loss_positions["position"], high_loss_positions["avg_loss"], 
                       color='red', s=50, alpha=0.5, label=f'High Loss (>{threshold:.2f})')
            plt.legend()
        
        output_file = "position_loss_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📊 位置Loss分析图已保存: {output_file}")
        plt.close()
    
    # Show statistics for key positions
    print("\n📍 关键位置统计:")
    print("-" * 80)
    
    # First token
    first_tokens = [r for r in records if r.get("position", 0) == 0]
    if first_tokens:
        first_loss = np.mean([r.get("gt_token_loss") or r.get("loss", 0.0) for r in first_tokens])
        print(f"位置 0 (第一个target token): avg_loss = {first_loss:.4f} (n={len(first_tokens)})")
    
    # Early positions (0-50)
    early = [r for r in records if 0 <= r.get("position", 0) < 50]
    if early:
        early_loss = np.mean([r.get("gt_token_loss") or r.get("loss", 0.0) for r in early])
        print(f"位置 0-50: avg_loss = {early_loss:.4f} (n={len(early)})")
    
    # Middle positions (50-150)
    middle = [r for r in records if 50 <= r.get("position", 0) < 150]
    if middle:
        middle_loss = np.mean([r.get("gt_token_loss") or r.get("loss", 0.0) for r in middle])
        print(f"位置 50-150: avg_loss = {middle_loss:.4f} (n={len(middle)})")
    
    # Late positions (150+)
    late = [r for r in records if r.get("position", 0) >= 150]
    if late:
        late_loss = np.mean([r.get("gt_token_loss") or r.get("loss", 0.0) for r in late])
        print(f"位置 150+: avg_loss = {late_loss:.4f} (n={len(late)})")
    
    return df


def analyze_topk_predictions(records: List[Dict[str, Any]], top_k: int = 20) -> pd.DataFrame:
    """Analyze top-k predictions and compare with ground truth."""
    print("\n" + "="*80)
    print("📊 第四步：Top-k预测对比分析")
    print("="*80)
    
    # Analyze prediction correctness and confidence
    analysis_results = []
    
    for record in records:
        # Support both old and new field names
        token_id = record.get("gt_token_id") or record.get("token_id", 0)
        token = record.get("gt_token") or record.get("token", "")
        loss = record.get("gt_token_loss") or record.get("loss", 0.0)
        is_correct = record.get("is_correct", False)
        topk = record.get("topk_predictions", [])
        
            if len(topk) > 0:
                top1_prob = topk[0].get("prob", 0.0)
                top1_token = topk[0].get("token", "")
                top1_token_id = topk[0].get("token_id", 0)
            
                # Check if correct token is in top-k
                correct_in_topk = any(pred.get("token_id", 0) == token_id for pred in topk)
                correct_rank = None
                if correct_in_topk:
                    for i, pred in enumerate(topk):
                        if pred.get("token_id", 0) == token_id:
                            correct_rank = i + 1
                            break
            
            analysis_results.append({
                "token_id": token_id,
                "token": token,
                "loss": loss,
                "is_correct": is_correct,
                "top1_prob": top1_prob,
                "top1_token": top1_token,
                "top1_token_id": top1_token_id,
                "correct_in_topk": correct_in_topk,
                "correct_rank": correct_rank,
            })
    
    df = pd.DataFrame(analysis_results)
    
    # Statistics
    total = len(df)
    correct_count = df["is_correct"].sum()
    correct_in_topk_count = df["correct_in_topk"].sum()
    
    print(f"\n📈 预测统计:")
    print("-" * 80)
    print(f"总Token数: {total}")
    print(f"Top-1正确: {correct_count} ({correct_count/total*100:.2f}%)")
    print(f"Top-5中包含正确答案: {correct_in_topk_count} ({correct_in_topk_count/total*100:.2f}%)")
    
    # Analyze high-loss but correct predictions (low confidence)
    high_loss_correct = df[(df["loss"] > df["loss"].quantile(0.9)) & (df["is_correct"] == True)]
    if len(high_loss_correct) > 0:
        print(f"\n⚠️ 高Loss但预测正确 (低置信度): {len(high_loss_correct)} tokens")
        print("   这些token虽然预测正确，但模型不够confident")
        print(f"   平均Top-1概率: {high_loss_correct['top1_prob'].mean():.4f}")
    
    # Analyze incorrect predictions
    incorrect = df[df["is_correct"] == False]
    if len(incorrect) > 0:
        print(f"\n❌ 预测错误: {len(incorrect)} tokens")
        print(f"   平均Loss: {incorrect['loss'].mean():.4f}")
        print(f"   平均Top-1概率: {incorrect['top1_prob'].mean():.4f}")
        
        # Show examples
        print("\n   示例错误预测 (Top 10 by loss):")
        top_errors = incorrect.nlargest(10, "loss")
        for _, row in top_errors.iterrows():
            print(f"     Token: {row['token']} (id={row['token_id']})")
            print(f"       Loss: {row['loss']:.4f}, Top-1: {row['top1_token']} (prob={row['top1_prob']:.4f})")
            print(f"       正确答案在Top-5中: {'是' if row['correct_in_topk'] else '否'} (排名: {row['correct_rank'] if row['correct_rank'] else 'N/A'})")
    
    return df


def generate_summary_report(
    high_loss_df: pd.DataFrame,
    type_df: pd.DataFrame,
    position_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    output_file: str = "token_loss_analysis_report.md"
):
    """Generate a comprehensive markdown report."""
    print("\n" + "="*80)
    print("📝 生成分析报告")
    print("="*80)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Token-level Loss Analysis Report\n\n")
        f.write("## 一、高频高Loss Token统计\n\n")
        f.write("### Top 20 高频高Loss Tokens\n\n")
        f.write(high_loss_df.head(20).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 二、Token类型聚类分析\n\n")
        f.write(type_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 三、位置敏感分析\n\n")
        f.write("### 关键位置统计\n\n")
        f.write("| 位置范围 | 平均Loss | Token数量 |\n")
        f.write("|---------|---------|----------|\n")
        
        # Calculate position ranges
        if len(position_df) > 0:
            early = position_df[position_df["position"] < 50]
            middle = position_df[(position_df["position"] >= 50) & (position_df["position"] < 150)]
            late = position_df[position_df["position"] >= 150]
            
            if len(early) > 0:
                f.write(f"| 0-50 | {early['avg_loss'].mean():.4f} | {early['count'].sum()} |\n")
            if len(middle) > 0:
                f.write(f"| 50-150 | {middle['avg_loss'].mean():.4f} | {middle['count'].sum()} |\n")
            if len(late) > 0:
                f.write(f"| 150+ | {late['avg_loss'].mean():.4f} | {late['count'].sum()} |\n")
        
        f.write("\n\n")
        
        f.write("## 四、Top-k预测对比分析\n\n")
        f.write(f"- 总Token数: {len(topk_df)}\n")
        f.write(f"- Top-1正确率: {topk_df['is_correct'].sum() / len(topk_df) * 100:.2f}%\n")
        f.write(f"- Top-5中包含正确答案: {topk_df['correct_in_topk'].sum() / len(topk_df) * 100:.2f}%\n")
        f.write("\n")
    
    print(f"✅ 分析报告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze token-level loss data")
    parser.add_argument(
        "--token_loss_dir",
        type=str,
        required=True,
        help="Directory containing token loss JSONL files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model for tokenizer (optional, for better token decoding)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for analysis results"
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
    high_loss_df = analyze_high_loss_tokens(records, top_k=50)
    type_df = analyze_token_type_clustering(records)
    position_df = analyze_position_sensitivity(records)
    topk_df = analyze_topk_predictions(records, top_k=20)
    
    # Generate report
    generate_summary_report(high_loss_df, type_df, position_df, topk_df)
    
    # Save detailed DataFrames
    high_loss_df.to_csv("high_loss_tokens.csv", index=False)
    type_df.to_csv("token_type_analysis.csv", index=False)
    position_df.to_csv("position_analysis.csv", index=False)
    topk_df.to_csv("topk_prediction_analysis.csv", index=False)
    
    print("\n✅ 所有分析完成！")
    print(f"📁 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
