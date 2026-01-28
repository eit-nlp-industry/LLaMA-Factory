#!/usr/bin/env python3
"""
在JSON文件的每个data项的system字段中添加retrieval_tool调用说明
"""

import json
import sys
import os

# retrieval_tool的调用说明内容
RETRIEVAL_TOOL_INSTRUCTIONS = """

**retrieval_tool调用格式说明**：
如果调用 retrieval_tool，只需输出查询参数，其他参数由系统自动补全：
<tool_call>
"query": "查询内容", "source_filter": "指定检索的知识库"
</tool_call>

retrieval_tool 示例：
<tool_call>
"query": "按时间范围查询订单，支持多种时间范围类型查询，包括最近天数、特定月份、日期范围、季度查询，提供趋势分析功能", "source_filter": "toollist"
</tool_call>

**注意**：其他工具（非 retrieval_tool）必须使用完整 JSON 格式。
"""


def add_retrieval_instructions(data):
    """递归处理数据，为每个system字段添加retrieval_tool说明"""
    
    if isinstance(data, dict):
        # 如果是字典，检查是否有system字段
        if 'system' in data and isinstance(data['system'], str):
            # 检查是否已经包含retrieval_tool说明
            if "**retrieval_tool调用格式说明**" not in data['system']:
                # 在规则1和规则2之间插入说明
                system_content = data['system']
                
                # 查找"1. 如果需要调用函数"的位置
                if '1.' in system_content and '2.' in system_content:
                    # 找到规则1和规则2之间的位置
                    parts = system_content.split('2.', 1)
                    if len(parts) == 2:
                        # 在规则2之前插入retrieval_tool说明
                        data['system'] = parts[0] + RETRIEVAL_TOOL_INSTRUCTIONS + '\n\n2.' + parts[1]
                elif '1.' in system_content:
                    # 如果没有规则2，在规则1之后插入
                    parts = system_content.split('1.', 1)
                    if len(parts) == 2:
                        # 找到规则1内容结束的位置（下一条规则或结束）
                        lines = parts[1].split('\n')
                        end_idx = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith(('2.', '3.', '4.', '5.')):
                                end_idx = i
                                break
                        if end_idx > 0:
                            inserted_content = '\n'.join(lines[:end_idx]) + '\n' + RETRIEVAL_TOOL_INSTRUCTIONS + '\n' + '\n'.join(lines[end_idx:])
                            data['system'] = parts[0] + '1.' + inserted_content
                        else:
                            data['system'] = system_content + RETRIEVAL_TOOL_INSTRUCTIONS
        
        # 递归处理字典中的所有值
        for k in data:
            if isinstance(data[k], (dict, list)):
                add_retrieval_instructions(data[k])
    
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        for item in data:
            add_retrieval_instructions(item)


def process_json_file(file_path):
    """处理JSON文件"""
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    add_retrieval_instructions(data)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已处理文件: {file_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python add_retrieval_tool_instructions.py <json_file1> [json_file2] ...")
        print("或: python add_retrieval_tool_instructions.py <directory>")
        sys.exit(1)
    
    paths = sys.argv[1:]
    
    for path in paths:
        if os.path.isfile(path) and path.endswith('.json'):
            process_json_file(path)
        elif os.path.isdir(path):
            # 如果是目录，处理目录下所有JSON文件
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        process_json_file(file_path)
        else:
            print(f"✗ 忽略非JSON文件: {path}")


if __name__ == '__main__':
    main()

