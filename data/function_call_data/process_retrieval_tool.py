#!/usr/bin/env python3
"""
合并功能：
1. 为包含特定function_call说明的system字段添加retrieval_tool调用说明
2. 简化JSON文件中retrieval_tool的function_call格式
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

# 需要检查的特定文本
REQUIRED_TEXT = "1. 如果需要调用函数，则 **只能输出一个函数调用**，格式如下：\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"


def should_add_instructions(system_content):
    """检查是否应该添加retrieval_tool说明"""
    # 检查是否包含特定的function_call说明
    if REQUIRED_TEXT not in system_content:
        return False
    
    # 检查是否已经包含retrieval_tool说明
    if "**retrieval_tool调用格式说明**" in system_content:
        return False
    
    return True


def add_retrieval_instructions(system_content):
    """在system字段中添加retrieval_tool说明"""
    # 查找"1. 如果需要调用函数"的位置
    if '1.' in system_content and '2.' in system_content:
        # 找到规则1和规则2之间的位置
        parts = system_content.split('2.', 1)
        if len(parts) == 2:
            # 在规则2之前插入retrieval_tool说明
            return parts[0] + RETRIEVAL_TOOL_INSTRUCTIONS + '\n\n2.' + parts[1]
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
                return parts[0] + '1.' + inserted_content
            else:
                return system_content + RETRIEVAL_TOOL_INSTRUCTIONS
    
    return system_content


def simplify_retrieval_tool_call(value_str):
    """简化retrieval_tool的函数调用格式"""
    try:
        # 尝试解析JSON
        call_data = json.loads(value_str)
        
        # 检查是否是retrieval_tool调用
        if isinstance(call_data, dict) and call_data.get('name') == 'retrieval_tool':
            arguments = call_data.get('arguments', {})
            query = arguments.get('query', '')
            source_filter = arguments.get('source_filter', '')
            
            # 返回简化格式
            return f'"query": "{query}", "source_filter": "{source_filter}"'
        
        # 如果不是retrieval_tool，返回原值
        return value_str
    except json.JSONDecodeError:
        # 如果不是有效的JSON，返回原值
        return value_str


def process_conversations(conversations):
    """处理conversations数组中的function_call"""
    if not isinstance(conversations, list):
        return
    
    for conv in conversations:
        if isinstance(conv, dict):
            # 如果是function_call类型，检查并简化
            if conv.get('from') == 'function_call' and 'value' in conv:
                old_value = conv['value']
                new_value = simplify_retrieval_tool_call(old_value)
                if new_value != old_value:
                    conv['value'] = new_value
                    print(f"  - 简化: {old_value[:100]}... -> {new_value[:100]}...")


def process_data(data):
    """递归处理数据"""
    if isinstance(data, dict):
        # 检查是否有conversations字段，处理function_call
        if 'conversations' in data:
            process_conversations(data['conversations'])
        
        # 检查并处理system字段
        if 'system' in data and isinstance(data['system'], str):
            if should_add_instructions(data['system']):
                old_system = data['system']
                data['system'] = add_retrieval_instructions(old_system)
                print(f"  - 添加retrieval_tool说明到system字段")
        
        # 递归处理字典中的所有值
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                process_data(value)
    
    elif isinstance(data, list):
        # 如果是列表，处理每个元素
        for item in data:
            process_data(item)


def process_json_file(file_path):
    """处理JSON文件"""
    print(f"处理文件: {file_path}")
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    process_data(data)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已完成: {file_path}\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python process_retrieval_tool.py <json_file1> [json_file2] ...")
        print("或: python process_retrieval_tool.py <directory>")
        sys.exit(1)
    
    paths = sys.argv[1:]
    
    for path in paths:
        # 转换为绝对路径
        abs_path = os.path.abspath(path)
        
        if os.path.isfile(abs_path) and abs_path.endswith('.json'):
            process_json_file(abs_path)
        elif os.path.isdir(abs_path):
            # 如果是目录，处理目录下所有JSON文件
            for root, dirs, files in os.walk(abs_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        process_json_file(file_path)
        else:
            print(f"✗ 文件或目录不存在: {path}")


if __name__ == '__main__':
    main()

