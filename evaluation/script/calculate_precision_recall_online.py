#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算tool_calls_comparison中precision和recall的脚本
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from loguru import logger
from utils.custom_logging import setup_logging
import os
from datetime import datetime

setup_logging()

def load_evaluation_results(file_path: str) -> Dict[str, Any]:
    """加载评估结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    """计算precision和recall"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def analyze_tool_calls_comparison(results: List[Dict]) -> Dict[str, Any]:
    """分析tool_calls_comparison数据"""
    
    # 存储统计信息
    stats = {
        'total_queries': len(results),
        'retrieval_tool': {
            'name': {'tp': 0, 'fp': 0, 'fn': 0},
            'arguments': {'tp': 0, 'fp': 0, 'fn': 0}
        },
        'non_retrieval_tool': {
            'name': {'tp': 0, 'fp': 0, 'fn': 0},
            'arguments': {'tp': 0, 'fp': 0, 'fn': 0}
        },
        'all_tools': {
            'name': {'tp': 0, 'fp': 0, 'fn': 0},
            'arguments': {'tp': 0, 'fp': 0, 'fn': 0}
        }
    }
    
    processed_queries = 0
    total_tool_calls = 0
    
    for result in results:
        if 'comparison' not in result or 'tool_calls_comparison' not in result['comparison']:
            continue
            
        processed_queries += 1
        comparison = result['comparison']['tool_calls_comparison']
        detailed_scores = comparison.get('detailed_scores', [])
        total_tool_calls += len(detailed_scores)
        
        # 分析每个tool call
        for score in detailed_scores:
            server_name = score.get('server_name', '')
            original_name = score.get('original_name', '')
            name_match = score.get('name_match', False)
            arguments_match = score.get('arguments_match', False)
            server_present = score.get('server_present', False)
            original_present = score.get('original_present', False)
            
            # 判断是否为retrieval tool
            is_retrieval = 'retrieval' in server_name.lower() or 'retrieval' in original_name.lower()
            
            # 选择统计类别
            tool_category = 'retrieval_tool' if is_retrieval else 'non_retrieval_tool'
            
            # 计算name的TP, FP, FN
            if name_match and server_present and original_present:
                # True Positive: 两个都存在且匹配
                stats[tool_category]['name']['tp'] += 1
                stats['all_tools']['name']['tp'] += 1
            elif server_present and original_present and not name_match:
                # False Positive: 两个都存在但不匹配
                stats[tool_category]['name']['fp'] += 1
                stats['all_tools']['name']['fp'] += 1
            elif server_present and not original_present:
                # False Positive: server有但original没有
                stats[tool_category]['name']['fp'] += 1
                stats['all_tools']['name']['fp'] += 1
            elif not server_present and original_present:
                # False Negative: original有但server没有
                stats[tool_category]['name']['fn'] += 1
                stats['all_tools']['name']['fn'] += 1
            
            # 计算arguments的TP, FP, FN
            if arguments_match and server_present and original_present:
                # True Positive: 两个都存在且匹配
                stats[tool_category]['arguments']['tp'] += 1
                stats['all_tools']['arguments']['tp'] += 1
            elif server_present and original_present and not arguments_match:
                # False Positive: 两个都存在但不匹配
                stats[tool_category]['arguments']['fp'] += 1
                stats['all_tools']['arguments']['fp'] += 1
            elif server_present and not original_present:
                # False Positive: server有但original没有
                stats[tool_category]['arguments']['fp'] += 1
                stats['all_tools']['arguments']['fp'] += 1
            elif not server_present and original_present:
                # False Negative: original有但server没有
                stats[tool_category]['arguments']['fn'] += 1
                stats['all_tools']['arguments']['fn'] += 1
    
    stats['processed_queries'] = processed_queries
    stats['total_tool_calls'] = total_tool_calls
    return stats

def print_results(stats: Dict[str, Any], output_file: str = None):
    """打印结果并保存到文件"""
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"precision_recall_results_{timestamp}.txt"
    
    # 准备输出内容
    output_lines = []
    
    def log_and_save(message: str):
        logger.info(message)
        output_lines.append(message)
    
    log_and_save("=" * 80)
    log_and_save("Tool Calls Comparison - Precision & Recall 分析结果")
    log_and_save("=" * 80)
    log_and_save(f"总查询数: {stats['total_queries']}")
    log_and_save(f"处理查询数: {stats['processed_queries']}")
    log_and_save(f"总Tool Calls数: {stats['total_tool_calls']}")
    log_and_save("")
    
    categories = ['retrieval_tool', 'non_retrieval_tool', 'all_tools']
    category_names = {
        'retrieval_tool': '包含Retrieval Tool',
        'non_retrieval_tool': '非Retrieval Tool', 
        'all_tools': '所有Tool'
    }
    
    for category in categories:
        log_and_save(f"【{category_names[category]}】")
        log_and_save("-" * 40)
        
        for metric_type in ['name', 'arguments']:
            metric_name = 'Tool Call Name' if metric_type == 'name' else 'Tool Call Arguments'
            tp = stats[category][metric_type]['tp']
            fp = stats[category][metric_type]['fp']
            fn = stats[category][metric_type]['fn']
            
            precision, recall = calculate_precision_recall(tp, fp, fn)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            log_and_save(f"{metric_name}:")
            log_and_save(f"  TP: {tp}, FP: {fp}, FN: {fn}")
            log_and_save(f"  Precision: {precision:.4f}")
            log_and_save(f"  Recall: {recall:.4f}")
            log_and_save(f"  F1-Score: {f1:.4f}")
            log_and_save("")
        log_and_save("")
    
    # 保存到文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"结果已保存到文件: {output_file}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")

def main():
    """主函数"""
    file_path = 'eval_results/evaluation_results.json'
    
    logger.info("正在加载评估结果...")
    data = load_evaluation_results(file_path)
    
    logger.info("正在分析tool_calls_comparison数据...")
    stats = analyze_tool_calls_comparison(data['results'])
    
    print_results(stats)

if __name__ == "__main__":
    main()
