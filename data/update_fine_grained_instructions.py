#!/usr/bin/env python3
"""
更新已有的细粒度数据到最新的instruction格式
简化instruction，使其更加清晰易懂
"""

import json
import os
from typing import Dict, List, Any
import argparse
import re

# 导入最新的资源结构定义
from generate_fine_grained_data import (
    RESOURCE_DETAIL_STRUCTURE, 
    get_field_description,
    get_all_resource_names
)

def create_updated_instruction(field_name: str, field_description: str, resource_name: str = None) -> str:
    """创建更新后的指令格式"""
    
    # 特殊处理resource_names字段
    if field_name == "resource_names":
        all_resource_names = get_all_resource_names()
        resource_list = "、".join(all_resource_names)
        return f"""请从OCR文本中抽取旅行订单中的所有资源名称。

可识别的资源名称包括：{resource_list}

资源别称表述：
- 大慈岩丛林速滑（旱滑道）下行：可能表述为"旱滑道"、"丛林速滑"
- 灵栖洞西游魔毯：可能表述为"飞天魔毯"、"灵栖洞魔毯"
- 灵栖洞极速滑道：可能表述为"灵栖洞滑道"、"灵栖洞速滑"
- 考拉森林丛林探险：可能表述为"考拉森林"、"丛林探险"
- 三江口游线（游船）：可能表述为"富春江游船"
- 严州古城：可能表述为"梅城"
- 三都渔村婚礼表演：可能表述为"三都渔村九姓渔氏水上婚礼表演"、"九姓渔氏"、"婚礼表演"

资源内在联系规则：
1. 七里扬帆景区：
   - 出现"七里扬帆"（无三江口）信息时，包含：七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船
   - 出现"三江口"信息时，包含：七里扬帆-三江口游线（游船），但不包含七里扬帆门票和七里扬帆游船
   - 出现"葫芦峡漂流"信息时，包含：七里扬帆-七里扬帆门票、七里扬帆-七里扬帆游船

2. 灵栖洞景区：
   - 出现"灵栖洞"信息时，包含：灵栖洞-灵栖洞门票、灵栖洞-灵栖洞手划船

严格按照以下JSON格式输出：
{{
  "resource_names": ["资源主体-资源名称1", "资源主体-资源名称2"] 或 []
}}"""
    
    # 特殊处理resource_detail字段
    if field_name == "resource_detail" and resource_name:
        detail_structure = RESOURCE_DETAIL_STRUCTURE.get(resource_name, {})
        if detail_structure:
            field_lines = []
            for field_type, enum_values in detail_structure.items():
                if enum_values:
                    enum_list = "、".join(f'"{str(v)}"' for v in enum_values)
                    field_lines.append(f"{field_type} ({get_field_description(field_type)}): {enum_list}")
                else:
                    field_lines.append(f"{field_type} ({get_field_description(field_type)}): null")
            
            fields_info = "\n".join(field_lines)
            
            # 构建JSON示例结构
            json_fields = []
            for field_type in detail_structure.keys():
                json_fields.append(f'    "{field_type}": 选择值或null')
            json_structure = ",\n".join(json_fields)
            
            return f"""请从OCR文本中抽取旅行订单中{resource_name}的详细信息。

提取字段及可选值：
{fields_info}

严格按照以下JSON格式输出：
{{
  "resource_detail": {{
{json_structure}
  }}
}}"""
        else:
            return f"""请从OCR文本中抽取旅行订单中{resource_name}的详细信息。

严格按照以下JSON格式输出：
{{
  "resource_detail": {{}} 
}}"""
    
    # 基础字段的指令定义
    field_instructions = {
        "team_size": """请从OCR文本中抽取旅行订单的总人数信息。

严格按照以下JSON格式输出：
{
  "team_size": 整数或null
}
注意：订单信息中的导游员、讲解员、司机、领队等人员，不包含在总人数中。
""",
        "start_date": """请从OCR文本中抽取旅行订单的开始日期信息。

严格按照以下JSON格式输出：
{
  "start_date": "YYYY-MM-DD"或null
}""",
        "end_date": """请从OCR文本中抽取旅行订单的结束日期信息。

严格按照以下JSON格式输出：
{
  "end_date": "YYYY-MM-DD"或null
}""",
        "payment_method": """请从OCR文本中抽取旅行订单的支付方式信息。

严格按照以下JSON格式输出：
{
  "payment_method": "支付方式"或null
}""",
        "customer_name": """请从OCR文本中抽取旅行订单的客户名称（通常是旅行社或公司名称）。

严格按照以下JSON格式输出：
{
  "customer_name": "客户名称"或null
}""",
        "customer_market": """请从OCR文本中抽取旅行订单的客户地区，按照'省-市'格式。

严格按照以下JSON格式输出：
{
  "customer_market": "省-市"或null
}""",
        "customer_type": """请从OCR文本中抽取旅行订单的客户类型（如旅行社、机构等）。

严格按照以下JSON格式输出：
{
  "customer_type": "客户类型"或null
}""",
        "notes": """请从OCR文本中抽取旅行订单的备注信息。

严格按照以下JSON格式输出：
{
  "notes": "备注信息"或null
}""",
        "contacts": """请从OCR文本中抽取旅行订单的联系人信息。

提取字段：
name (联系人姓名)
phone (联系电话)
idcard (身份证号码，如果没有则为null)

严格按照以下JSON格式输出：
{
  "contacts": {
    "data": [
      {
        "name": "姓名",
        "phone": "电话", 
        "idcard": "身份证号"或null
      }
    ]
  }
}""",
        "resource_start_time": f"""请从OCR文本中抽取旅行订单中{resource_name or '该资源'}的开始时间。

严格按照以下JSON格式输出：
{{
  "resource_start_time": "YYYY-MM-DD"或null
}}""",
        "resource_end_time": f"""请从OCR文本中抽取旅行订单中{resource_name or '该资源'}的结束时间。

严格按照以下JSON格式输出：
{{
  "resource_end_time": "YYYY-MM-DD"或null
}}""",
        "resource_team_size": f"""请从OCR文本中抽取旅行订单中{resource_name or '该资源'}的使用人数。

严格按照以下JSON格式输出：
{{
  "resource_team_size": 整数或null
}}
注意：订单信息中的导游员、讲解员、司机、领队等人员，不包含在总人数中。
"""
    }
    
    return field_instructions.get(field_name, f"""请从OCR文本中抽取旅行订单的{field_description}信息。

严格按照以下JSON格式输出：
{{
  "{field_name}": "值"或null
}}""")

def analyze_old_instruction(instruction: str) -> Dict[str, str]:
    """分析旧的instruction，提取字段信息"""
    result = {
        "field_name": None,
        "field_description": None,
        "resource_name": None
    }
    
    # 分析输出格式来确定字段名
    if '"team_size"' in instruction:
        result["field_name"] = "team_size"
        result["field_description"] = "总人数"
    elif '"start_date"' in instruction:
        result["field_name"] = "start_date"
        result["field_description"] = "开始日期"
    elif '"end_date"' in instruction:
        result["field_name"] = "end_date"
        result["field_description"] = "结束日期"
    elif '"payment_method"' in instruction:
        result["field_name"] = "payment_method"
        result["field_description"] = "支付方式"
    elif '"customer_name"' in instruction:
        result["field_name"] = "customer_name"
        result["field_description"] = "客户名称"
    elif '"customer_market"' in instruction:
        result["field_name"] = "customer_market"
        result["field_description"] = "客户地区"
    elif '"customer_type"' in instruction:
        result["field_name"] = "customer_type"
        result["field_description"] = "客户类型"
    elif '"notes"' in instruction:
        result["field_name"] = "notes"
        result["field_description"] = "备注"
    elif '"contacts"' in instruction:
        result["field_name"] = "contacts"
        result["field_description"] = "联系人信息"
    elif '"resource_names"' in instruction:
        result["field_name"] = "resource_names"
        result["field_description"] = "资源名称列表"
    elif '"resource_start_time"' in instruction:
        result["field_name"] = "resource_start_time"
        result["field_description"] = "开始时间"
    elif '"resource_end_time"' in instruction:
        result["field_name"] = "resource_end_time"
        result["field_description"] = "结束时间"
    elif '"resource_team_size"' in instruction:
        result["field_name"] = "resource_team_size"
        result["field_description"] = "使用人数"
    elif '"resource_detail"' in instruction:
        result["field_name"] = "resource_detail"
        result["field_description"] = "详细信息"
    
    # 提取资源名称（如果有的话）
    if "请从OCR文本中抽取旅行订单中" in instruction and result["field_name"] in ["resource_start_time", "resource_end_time", "resource_team_size", "resource_detail"]:
        # 尝试从instruction中提取资源名称
        lines = instruction.split('\n')
        for line in lines:
            if "请从OCR文本中抽取旅行订单中" in line and "的" in line:
                # 找到资源名称
                start_idx = line.find("请从OCR文本中抽取旅行订单中") + len("请从OCR文本中抽取旅行订单中")
                end_idx = line.find("的", start_idx)
                if end_idx > start_idx:
                    resource_part = line[start_idx:end_idx]
                    result["resource_name"] = resource_part
                    break
    
    return result

def parse_old_output(output: str, field_name: str) -> Any:
    """解析旧的output格式，提取实际值"""
    if not output:
        return None
    
    # 1) 优先按整体JSON解析（适配例如: "{"team_size": 36}" 这种字符串）
    try:
        parsed = json.loads(output) if isinstance(output, str) else output
        if isinstance(parsed, dict) and field_name in parsed:
            return parsed[field_name]
    except Exception:
        pass

    # 2) 回退到基于键名的提取逻辑，尽量健壮地抽取值片段
    if isinstance(output, str) and f'"{field_name}"' in output:
        # 使用正则尝试获取字段值的原始JSON片段
        # 匹配 null/布尔/数字/字符串/对象/数组
        pattern = rf'"{re.escape(field_name)}"\s*:\s*(null|true|false|-?\d+(?:\.\d+)?|"[^"\\]*(?:\\.[^"\\]*)*"|\{{[^\}}]*\}}|\[[^\]]*\])'
        match = re.search(pattern, output, flags=re.IGNORECASE | re.DOTALL)
        if match:
            raw_value = match.group(1).strip()
            # 先尝试按JSON解析
            try:
                return json.loads(raw_value)
            except Exception:
                # 处理裸值
                if raw_value == 'null':
                    return None
                if raw_value.startswith('"') and raw_value.endswith('"'):
                    return raw_value[1:-1]
                # 数字
                try:
                    return int(raw_value)
                except Exception:
                    try:
                        return float(raw_value)
                    except Exception:
                        return raw_value

    return None

def format_new_output(value: Any, field_name: str) -> str:
    """格式化新的output为标准JSON格式"""
    result = {field_name: value}
    return json.dumps(result, ensure_ascii=False)

def update_fine_grained_data(input_file: str, output_file: str):
    """更新细粒度数据到最新格式"""
    print(f"正在加载数据: {input_file}")
    
    # 加载数据 - 支持JSON数组格式和JSONL格式
    data = []
    is_json_array_format = False  # 记录输入格式类型
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # 检测文件格式
        if content.startswith('[') and content.endswith(']'):
            # JSON数组格式
            try:
                data = json.loads(content)
                is_json_array_format = True
                print(f"检测到JSON数组格式")
            except json.JSONDecodeError as e:
                print(f"JSON数组解析失败: {e}")
                return
        else:
            # JSONL格式 - 按行解析
            print(f"检测到JSONL格式")
            f.seek(0)  # 重置文件指针
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
    
    print(f"加载了 {len(data)} 条数据")
    
    # 更新数据
    updated_data = []
    update_stats = {}
    
    for item in data:
        old_instruction = item["instruction"]
        old_output = item["output"]
        
        # 分析旧instruction
        analysis = analyze_old_instruction(old_instruction)
        
        if analysis["field_name"]:
            # 生成新instruction
            new_instruction = create_updated_instruction(
                analysis["field_name"], 
                analysis["field_description"], 
                analysis["resource_name"]
            )
            
            # 解析并转换output格式
            parsed_value = parse_old_output(old_output, analysis["field_name"])
            new_output = format_new_output(parsed_value, analysis["field_name"])
            
            # 更新数据项：在原条目基础上修改，保留未知字段（例如 context）
            updated_item = dict(item)
            updated_item["instruction"] = new_instruction
            updated_item["output"] = new_output
            
            updated_data.append(updated_item)
            
            # 统计更新情况
            field_key = analysis["field_name"]
            if analysis["resource_name"]:
                field_key = f"{analysis['field_name']}({analysis['resource_name']})"
            update_stats[field_key] = update_stats.get(field_key, 0) + 1
        else:
            # 无法识别的格式，保持原样
            updated_data.append(item)
            update_stats["未识别"] = update_stats.get("未识别", 0) + 1
    
    # 保存更新后的数据 - 根据输入格式选择输出格式
    print(f"正在保存更新后的数据: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        if is_json_array_format:
            # 输出JSON数组格式
            print("保存为JSON数组格式")
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        else:
            # 输出JSONL格式
            print("保存为JSONL格式")
            for item in updated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    print(f"\n更新完成！共处理 {len(updated_data)} 条数据")
    print("\n各字段更新统计:")
    for field, count in sorted(update_stats.items()):
        print(f"  {field}: {count} 条")

def main():
    parser = argparse.ArgumentParser(description='更新细粒度数据到最新instruction格式')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        update_fine_grained_data(args.input, args.output)
    except Exception as e:
        print(f"更新失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
