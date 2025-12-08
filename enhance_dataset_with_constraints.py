#!/usr/bin/env python3
"""
数据增强脚本
将增强的系统提示应用到训练数据，并添加约束验证
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load_enhanced_system_prompt():
    """加载增强的系统提示"""
    prompt_path = "enhanced_system_prompt.txt"
    if Path(prompt_path).exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        # 如果文件不存在，使用默认的增强提示
        return """
# 工具调用规则

你是一个专业的工具调用助手。在处理用户查询时，必须严格遵循以下流程和规则：

## 第一阶段：检索工具调用（必须执行）

**规则1：检索阶段强制约束**
- 当用户提出查询时，**必须首先调用 retrieval_tool** 来检索相关工具
- retrieval_tool 的参数要求：
  - `query`: 必须使用用户的完整查询内容
  - `source_filter`: 必须设置为 "toollist"（用于检索工具库）
  - `top_k`: 建议设置为 5，以获取最相关的工具列表
  - `user_id`: 必须提供有效的用户ID

## 第二阶段：业务工具选择（严格约束）

**规则2：工具选择约束**
- 从 retrieval_tool 返回的 **top5 工具列表** 中选择最合适的业务工具
- 必须从返回的工具列表中选择，**不能调用列表外的工具**

## 第三阶段：参数提取（严格遵循）

**规则3：参数提取约束**
- 必须严格按照工具的 inputSchema 定义提取参数
- 对于必填参数（required字段），必须全部提供
- 参数类型必须匹配 inputSchema 的定义

## 第四阶段：结果总结（准确完整）

**规则4：总结生成要求**
- 基于工具返回的结果，生成准确、完整的总结
- 如果工具返回空数据，明确说明原因

记住：严格按照流程执行，确保每一步都符合约束要求。
"""

def validate_and_enhance_sample(sample: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    验证并增强单个样本
    返回: (是否有效, 错误信息, 增强后的样本)
    """
    conversations = sample.get("conversations", [])
    
    # 检查基本结构
    if not conversations:
        return False, "缺少conversations字段", sample
    
    # 检查是否包含retrieval_tool调用
    has_retrieval = False
    retrieval_idx = -1
    for i, msg in enumerate(conversations):
        if msg.get("from") == "function_call":
            try:
                value_str = msg.get("value", "{}")
                if isinstance(value_str, str):
                    value = json.loads(value_str)
                else:
                    value = value_str
                
                if value.get("name") == "retrieval_tool":
                    has_retrieval = True
                    retrieval_idx = i
                    # 验证参数
                    args = value.get("arguments", {})
                    if not isinstance(args, dict):
                        return False, f"retrieval_tool参数格式错误 (索引 {i})", sample
                    if "query" not in args:
                        return False, f"retrieval_tool缺少query参数 (索引 {i})", sample
                    if args.get("source_filter") != "toollist":
                        # 自动修复
                        args["source_filter"] = "toollist"
                        value["arguments"] = args
                        if isinstance(msg.get("value"), str):
                            msg["value"] = json.dumps(value, ensure_ascii=False)
                        else:
                            msg["value"] = value
            except json.JSONDecodeError:
                return False, f"function_call JSON解析失败 (索引 {i})", sample
            except Exception as e:
                return False, f"处理function_call时出错 (索引 {i}): {str(e)}", sample
    
    if not has_retrieval:
        return False, "缺少retrieval_tool调用", sample
    
    # 检查工具选择
    if retrieval_idx + 1 < len(conversations):
        observation = conversations[retrieval_idx + 1]
        if observation.get("from") == "observation":
            try:
                obs_value = observation.get("value", "[]")
                if isinstance(obs_value, str):
                    tool_list = json.loads(obs_value)
                else:
                    tool_list = obs_value
                
                if not isinstance(tool_list, list):
                    return False, "retrieval_tool返回结果格式错误", sample
                
                tool_names = [tool.get("name") for tool in tool_list if isinstance(tool, dict) and "name" in tool]
                
                # 检查后续的业务工具调用
                for i in range(retrieval_idx + 2, len(conversations)):
                    msg = conversations[i]
                    if msg.get("from") == "function_call":
                        try:
                            value_str = msg.get("value", "{}")
                            if isinstance(value_str, str):
                                value = json.loads(value_str)
                            else:
                                value = value_str
                            
                            tool_name = value.get("name")
                            if tool_name and tool_name != "retrieval_tool":
                                if tool_name not in tool_names:
                                    return False, f"业务工具 {tool_name} 不在retrieval_tool返回的列表中", sample
                        except:
                            pass
            except json.JSONDecodeError:
                return False, "observation JSON解析失败", sample
            except Exception as e:
                return False, f"验证工具选择时出错: {str(e)}", sample
    
    # 应用增强的系统提示
    enhanced_sample = sample.copy()
    enhanced_sample["system"] = load_enhanced_system_prompt()
    
    return True, "", enhanced_sample

def enhance_dataset(input_path: str, output_path: str, strict: bool = False):
    """
    增强数据集
    
    Args:
        input_path: 输入数据文件路径
        output_path: 输出数据文件路径
        strict: 是否严格模式（只保留完全符合约束的样本）
    """
    print(f"📖 读取数据文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 原始样本数: {len(data)}")
    
    enhanced_data = []
    errors = []
    fixed_count = 0
    
    for idx, sample in enumerate(data):
        is_valid, error_msg, enhanced_sample = validate_and_enhance_sample(sample)
        
        if is_valid:
            enhanced_data.append(enhanced_sample)
        else:
            if strict:
                errors.append(f"样本 {idx}: {error_msg}")
            else:
                # 非严格模式：尝试修复后仍添加
                enhanced_data.append(enhanced_sample)
                if error_msg:
                    fixed_count += 1
                    print(f"   ⚠️  样本 {idx} 已修复: {error_msg}")
    
    if errors:
        print(f"\\n❌ 发现 {len(errors)} 个错误（严格模式）:")
        for error in errors[:20]:  # 只显示前20个
            print(f"   {error}")
        if len(errors) > 20:
            print(f"   ... 还有 {len(errors) - 20} 个错误")
    
    print(f"\\n✅ 增强后样本数: {len(enhanced_data)}")
    if fixed_count > 0:
        print(f"   🔧 修复样本数: {fixed_count}")
    
    # 保存增强后的数据
    print(f"\\n💾 保存到: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 完成！")
    return len(enhanced_data), len(errors)

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python enhance_dataset_with_constraints.py <输入文件> <输出文件> [--strict]")
        print("示例: python enhance_dataset_with_constraints.py data/dataset/12_08/train.json data/dataset/12_08/train_enhanced.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    strict = "--strict" in sys.argv
    
    if not Path(input_path).exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    print("🚀 开始增强数据集")
    print("=" * 60)
    
    enhance_dataset(input_path, output_path, strict)
    
    print("\\n" + "=" * 60)
    print("✅ 数据增强完成！")

if __name__ == "__main__":
    main()

