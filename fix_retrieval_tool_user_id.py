#!/usr/bin/env python3
"""
修复数据集中 retrieval_tool 调用缺少 user_id 的问题
从样本顶层读取 user_id，并添加到 retrieval_tool 的 arguments 中
"""

import json
import sys
from pathlib import Path


def fix_retrieval_tool_user_id(data_file):
    """修复数据文件中 retrieval_tool 调用缺少 user_id 的问题"""
    
    print(f"🔧 修复文件: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_fixed = 0
    total_samples = len(data)
    
    for idx, item in enumerate(data):
        # 获取样本顶层的 user_id
        sample_user_id = item.get("user_id")
        if sample_user_id is None:
            print(f"⚠️  样本 {idx}: 顶层缺少 user_id，跳过")
            continue
        
        conversations = item.get("conversations", [])
        fixed_in_sample = False
        
        for i, msg in enumerate(conversations):
            if msg.get("from") == "function_call":
                try:
                    value_str = msg.get("value", "{}")
                    value = json.loads(value_str)
                    
                    if value.get("name") == "retrieval_tool":
                        args = value.get("arguments", {})
                        
                        # 检查是否缺少 user_id
                        if "user_id" not in args:
                            # 添加 user_id
                            args["user_id"] = sample_user_id
                            value["arguments"] = args
                            
                            # 更新消息
                            msg["value"] = json.dumps(value, ensure_ascii=False)
                            fixed_in_sample = True
                            total_fixed += 1
                            
                except json.JSONDecodeError as e:
                    print(f"⚠️  样本 {idx}, 消息 {i}: JSON 解析失败: {e}")
                except Exception as e:
                    print(f"⚠️  样本 {idx}, 消息 {i}: 处理失败: {e}")
        
        if fixed_in_sample:
            # 更新数据项
            item["conversations"] = conversations
    
    # 保存修复后的数据
    if total_fixed > 0:
        backup_file = str(data_file) + ".backup"
        print(f"📦 创建备份: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 保存修复后的数据...")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 修复完成:")
        print(f"   - 总样本数: {total_samples}")
        print(f"   - 修复的 retrieval_tool 调用: {total_fixed}")
        print(f"   - 备份文件: {backup_file}")
    else:
        print(f"ℹ️  没有需要修复的内容（所有 retrieval_tool 调用都已包含 user_id）")
    
    return total_fixed


def main():
    if len(sys.argv) < 2:
        print("用法: python fix_retrieval_tool_user_id.py <数据文件路径>")
        print("示例: python fix_retrieval_tool_user_id.py data/dataset/12_08/train.json")
        sys.exit(1)
    
    data_file = Path(sys.argv[1])
    if not data_file.exists():
        print(f"❌ 文件不存在: {data_file}")
        sys.exit(1)
    
    try:
        fixed_count = fix_retrieval_tool_user_id(data_file)
        sys.exit(0 if fixed_count >= 0 else 1)
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

