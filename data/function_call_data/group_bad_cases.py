#!/usr/bin/env python3
"""
将两个JSON文件中的bad case按照target_tool_name进行分组
"""
import json
from collections import defaultdict
from pathlib import Path

def load_json_file(filepath):
    """加载JSON文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_bad_cases_by_tool(precision_file, arg_file, output_file):
    """按照target_tool_name对bad case进行分组"""
    
    # 读取两个文件
    precision_data = load_json_file(precision_file)
    arg_data = load_json_file(arg_file)
    
    # 按照target_tool_name分组
    grouped_cases = defaultdict(lambda: {
        'precision_failed': [],
        'arg_failed': [],
        'total_count': 0
    })
    
    # 处理precision失败的cases
    for case in precision_data.get('cases', []):
        tool_name = case.get('target_tool_name', 'unknown')
        grouped_cases[tool_name]['precision_failed'].append(case)
        grouped_cases[tool_name]['total_count'] += 1
    
    # 处理argument失败的cases
    for case in arg_data.get('cases', []):
        tool_name = case.get('target_tool_name', 'unknown')
        grouped_cases[tool_name]['arg_failed'].append(case)
        grouped_cases[tool_name]['total_count'] += 1
    
    # 按照total_count排序
    sorted_groups = sorted(
        grouped_cases.items(), 
        key=lambda x: x[1]['total_count'], 
        reverse=True
    )
    
    # 构建输出结果
    result = {
        'summary': {
            'precision_file': str(precision_file),
            'arg_file': str(arg_file),
            'precision_total_failed': len(precision_data.get('cases', [])),
            'arg_total_failed': len(arg_data.get('cases', [])),
            'total_unique_tools': len(grouped_cases),
            'precision_summary': precision_data.get('summary', {}),
            'arg_summary': arg_data.get('summary', {})
        },
        'grouped_by_tool': {}
    }
    
    # 添加每个工具的分组数据
    for tool_name, data in sorted_groups:
        result['grouped_by_tool'][tool_name] = {
            'tool_name': tool_name,
            'total_failed_count': data['total_count'],
            'precision_failed_count': len(data['precision_failed']),
            'arg_failed_count': len(data['arg_failed']),
            'precision_failed_cases': data['precision_failed'],
            'arg_failed_cases': data['arg_failed']
        }
    
    # 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"分组完成！")
    print(f"=" * 80)
    print(f"Precision失败总数: {result['summary']['precision_total_failed']}")
    print(f"Argument失败总数: {result['summary']['arg_total_failed']}")
    print(f"涉及的工具总数: {result['summary']['total_unique_tools']}")
    print(f"=" * 80)
    print(f"\n按target_tool_name分组统计:")
    print(f"{'工具名称':<40} {'Precision失败':<15} {'Argument失败':<15} {'总计':<10}")
    print(f"-" * 80)
    
    for tool_name, data in sorted_groups:
        precision_count = len(data['precision_failed'])
        arg_count = len(data['arg_failed'])
        total = data['total_count']
        print(f"{tool_name:<40} {precision_count:<15} {arg_count:<15} {total:<10}")
    
    print(f"=" * 80)
    print(f"结果已保存到: {output_file}")

if __name__ == '__main__':
    precision_file = '/home/ziqiang/LLaMA-Factory/data/dataset/11_15/eval/data_train_prod_test_precision_failed.json'
    arg_file = '/home/ziqiang/LLaMA-Factory/data/dataset/11_15/eval/data_train_prod_test_arg_failed.json'
    output_file = '/home/ziqiang/LLaMA-Factory/data/dataset/11_15/grouped_bad_cases_by_tool.json'
    
    group_bad_cases_by_tool(precision_file, arg_file, output_file)

