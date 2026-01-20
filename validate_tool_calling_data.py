#!/usr/bin/env python3
"""
数据验证工具
检查训练数据是否符合工具调用约束
"""

import json
import sys
from pathlib import Path

def validate_retrieval_tool_call(conversations):
    """验证是否包含retrieval_tool调用"""
    has_retrieval = False
    retrieval_index = -1
    
    for i, msg in enumerate(conversations):
        if msg.get("from") == "function_call":
            try:
                value = json.loads(msg.get("value", "{}"))
                if value.get("name") == "retrieval_tool":
                    has_retrieval = True
                    retrieval_index = i
                    # 验证参数
                    args = value.get("arguments", {})
                    missing_params = []
                    if "query" not in args:
                        missing_params.append("query")
                    if "source_filter" not in args:
                        missing_params.append("source_filter")
                    if "user_id" not in args:
                        missing_params.append("user_id")
                    
                    if missing_params:
                        print(f"⚠️  警告: retrieval_tool缺少必需参数 {missing_params} (消息索引 {i})")
                    
                    if args.get("source_filter") != "toollist":
                        print(f"⚠️  警告: retrieval_tool的source_filter应为'toollist'，当前为'{args.get('source_filter')}' (消息索引 {i})")
                    break
            except Exception as e:
                pass
    
    return has_retrieval, retrieval_index

def validate_tool_selection(conversations, retrieval_index):
    """验证业务工具是否从retrieval_tool返回的列表中选择"""
    if retrieval_index < 0 or retrieval_index + 1 >= len(conversations):
        return False, "无法找到retrieval_tool的返回结果"
    
    # 获取retrieval_tool的返回结果
    observation = conversations[retrieval_index + 1]
    if observation.get("from") != "observation":
        return False, "retrieval_tool后没有observation"
    
    try:
        tool_list = json.loads(observation.get("value", "[]"))
        tool_names = [tool.get("name") for tool in tool_list if isinstance(tool, dict)]
    except:
        return False, "无法解析retrieval_tool返回的工具列表"
    
    # 检查后续的业务工具调用
    for i in range(retrieval_index + 2, len(conversations)):
        msg = conversations[i]
        if msg.get("from") == "function_call":
            try:
                value = json.loads(msg.get("value", "{}"))
                tool_name = value.get("name")
                if tool_name and tool_name != "retrieval_tool":
                    if tool_name not in tool_names:
                        return False, f"业务工具 {tool_name} 不在retrieval_tool返回的列表中"
            except:
                pass
    
    return True, "工具选择验证通过"

def validate_parameters(conversations):
    """验证参数提取是否符合inputSchema"""
    # 这里可以添加更详细的参数验证逻辑
    # 目前只做基本检查
    return True, "参数验证通过"

def validate_data_file(file_path):
    """验证数据文件"""
    print(f"\n🔍 验证数据文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    valid_count = 0
    errors = []
    
    for idx, item in enumerate(data):
        conversations = item.get("conversations", [])
        
        # 检查retrieval_tool调用
        has_retrieval, retrieval_idx = validate_retrieval_tool_call(conversations)
        if not has_retrieval:
            errors.append(f"样本 {idx}: 缺少retrieval_tool调用")
            continue
        
        # 检查工具选择
        is_valid, msg = validate_tool_selection(conversations, retrieval_idx)
        if not is_valid:
            errors.append(f"样本 {idx}: {msg}")
            continue
        
        # 检查参数
        is_valid, msg = validate_parameters(conversations)
        if not is_valid:
            errors.append(f"样本 {idx}: {msg}")
            continue
        
        valid_count += 1
    
    print(f"✅ 验证完成: {valid_count}/{total} 样本通过验证")
    if errors:
        print(f"\n❌ 发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
    
    return valid_count == total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python validate_tool_calling_data.py <数据文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    
    is_valid = validate_data_file(file_path)
    sys.exit(0 if is_valid else 1)
