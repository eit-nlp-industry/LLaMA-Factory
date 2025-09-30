#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按字段分类评估预测结果的准确率
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def extract_field_from_instruction(instruction: str) -> str:
    """从instruction中提取字段名"""
    # 新增的字段分类
    # 1. service_name 类 - 服务类型识别
    if 'service_name' in instruction and ('服务类型' in instruction or '识别服务类型' in instruction):
        return 'service_name'
    
    # 2. node_name 类 - 节点名称识别
    if 'node_name' in instruction and ('price_conditions' in instruction or '识别节点' in instruction):
        return 'node_name'
    
    # 3. re-rank 类 - API重排序
    if 're-rank' in instruction or 're-rank' in instruction.lower():
        return 're_rank'
    
    # 4. 最终回答生成类 - 根据用户的查询(input)和工具调用返回结果(context)生成最终回答
    normalized_instruction = instruction.replace('\n', ' ').replace('  ', ' ').strip()
    if '请根据以上用户的查询(input)和工具调用返回结果(context)' in normalized_instruction:
        return 'final_answer'
    
    # 5. 用户意图分析类 - 根据query和工具信息分析用户意图
    if '请根据query和工具信息，分析用户意图' in normalized_instruction:
        return 'user_intent_analysis'
    
    # 调试：检查指令内容
    if '请根据以上用户的查询' in instruction or '请根据query和工具信息' in instruction:
        print(f"调试 - 指令包含关键词但未匹配: {instruction[:200]}...")
        print(f"调试 - 标准化后: {normalized_instruction[:200]}...")
    
    # 6. 价格政策计算类 - 根据价格政策信息和OCR文本提取并计算资源价格
    if '根据以下价格政策信息和OCR文本，提取并计算资源价格' in instruction:
        return 'price_policy_calculation'
    
    # 资源明细类任务：指令不直接出现 resource_detail，但会出现其子键名
    resource_detail_keys = [
        'ship_combo', 'ship_items',
        'ticket_combo', 'ticket_items',
        'guide_items',
        'meal_standard', 'meal_standard_table',
    ]

    # 提取JSON格式中的字段名
    json_pattern = r'"([^"]+)"\s*:'
    matches = re.findall(json_pattern, instruction)
    
    if matches:
        # 过滤掉一些常见的非字段名
        filtered = [m for m in matches if m not in ['YYYY-MM-DD', '资源主体-资源名称1', '资源主体-资源名称2']]
        if filtered:
            # 如果匹配列表中包含 resource_detail 的子键，则优先返回 resource_detail
            if any(m in resource_detail_keys for m in filtered):
                return 'resource_detail'
            return filtered[0]
    
    # 如果正则未匹配，但字符串中包含 resource_detail 子键，也返回 resource_detail
    if any(key in instruction for key in resource_detail_keys):
        return 'resource_detail'

    # 如果没找到，尝试其他模式
    if 'team_size' in instruction:
        return 'team_size'
    elif 'end_date' in instruction:
        return 'end_date'
    elif 'start_date' in instruction:
        return 'start_date'
    elif 'customer_name' in instruction:
        return 'customer_name'
    elif 'resource_names' in instruction:
        return 'resource_names'
    elif 'resource_name' in instruction:
        return 'resource_name'
    elif 'resource_start_time' in instruction:
        return 'resource_start_time'
    elif 'resource_end_time' in instruction:
        return 'resource_end_time'
    elif 'resource_team_size' in instruction:
        return 'resource_team_size'
    elif 'resource_detail' in instruction:
        return 'resource_detail'
    
    return 'unknown'


def extract_json_from_response(response: str) -> Any:
    """从响应中提取JSON"""
    try:
        # 移除<think>标签和其他非JSON内容
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # 尝试解析JSON
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 如果解析失败，尝试提取大括号内的内容
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None


def _extract_balanced_json_object(text: str) -> str:
    """从字符串中提取首个平衡的大括号JSON对象子串。失败返回空串。"""
    if not isinstance(text, str):
        return ""
    start_index = text.find('{')
    if start_index == -1:
        return ""
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]
    return ""


def _parse_possible_json_string(value: Any) -> Any:
    """宽松解析：value可能是JSON字符串，可能多了尾随的大括号/噪声，尽力解析为对象。"""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    # 直接尝试解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 提取平衡的JSON对象子串再解析
    candidate = _extract_balanced_json_object(s)
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return value


def normalize_value(value: Any) -> str:
    """标准化值用于比较"""
    if value is None:
        return "null"
    elif isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        return str(value)


def smart_compare_json(predict_json: Any, label_json: Any, field_name: str) -> bool:
    """智能比较JSON值，处理特殊字段的比较逻辑"""
    
    # 新增字段：都需要完全准确率
    # 1. service_name - 服务类型识别
    if field_name == 'service_name':
        return compare_service_name(predict_json, label_json)
    
    # 2. node_name - 节点名称识别
    if field_name == 'node_name':
        return compare_node_name(predict_json, label_json)
    
    # 3. re_rank - API重排序
    if field_name == 're_rank':
        return compare_re_rank(predict_json, label_json)
    
    # 4. final_answer - 最终回答生成
    if field_name == 'final_answer':
        return compare_final_answer(predict_json, label_json)
    
    # 5. user_intent_analysis - 用户意图分析
    if field_name == 'user_intent_analysis':
        return compare_user_intent_analysis(predict_json, label_json)
    
    # 6. price_policy_calculation - 价格政策计算
    if field_name == 'price_policy_calculation':
        return compare_price_policy_calculation(predict_json, label_json)
    
    # 对于contacts字段，忽略数组内元素的顺序
    if field_name == 'contacts':
        return compare_contacts(predict_json, label_json)
    
    # 对于resource_names字段，忽略数组元素的顺序
    if field_name == 'resource_names':
        return compare_resource_names(predict_json, label_json)
    
    # 对于resource_name字段，支持单个资源名称或资源名称数组
    if field_name == 'resource_name':
        return compare_resource_name(predict_json, label_json)
    
    # 对于resource_detail字段，处理null值的特殊逻辑
    if field_name == 'resource_detail':
        return compare_resource_detail(predict_json, label_json)
    
    # 其他字段使用标准比较
    return normalize_value(predict_json) == normalize_value(label_json)


def compare_contacts(predict_json: Any, label_json: Any) -> bool:
    """比较contacts字段，忽略数组内元素的顺序"""
    try:
        # 解析contacts字段中的JSON字符串
        if isinstance(predict_json, dict) and 'contacts' in predict_json:
            predict_data = json.loads(predict_json['contacts'])
        else:
            predict_data = predict_json
            
        if isinstance(label_json, dict) and 'contacts' in label_json:
            label_data = json.loads(label_json['contacts'])
        else:
            label_data = label_json
        
        # 如果都有data字段，比较其中的数组（忽略顺序）
        if (isinstance(predict_data, dict) and 'data' in predict_data and 
            isinstance(label_data, dict) and 'data' in label_data):
            
            predict_list = predict_data['data']
            label_list = label_data['data']
            
            if len(predict_list) != len(label_list):
                return False
            
            # 将每个联系人转换为可比较的格式并排序
            predict_sorted = sorted([json.dumps(item, ensure_ascii=False, sort_keys=True) 
                                   for item in predict_list])
            label_sorted = sorted([json.dumps(item, ensure_ascii=False, sort_keys=True) 
                                 for item in label_list])
            
            return predict_sorted == label_sorted
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, KeyError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_resource_names(predict_json: Any, label_json: Any) -> bool:
    """比较resource_names字段，忽略数组元素的顺序"""
    try:
        # 解析resource_names字段中的JSON数组字符串
        if isinstance(predict_json, dict) and 'resource_names' in predict_json:
            predict_list = json.loads(predict_json['resource_names'])
        else:
            predict_list = predict_json
            
        if isinstance(label_json, dict) and 'resource_names' in label_json:
            label_list = json.loads(label_json['resource_names'])
        else:
            label_list = label_json
        
        # 如果都是列表，比较排序后的结果
        if isinstance(predict_list, list) and isinstance(label_list, list):
            return sorted(predict_list) == sorted(label_list)
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_resource_name(predict_json: Any, label_json: Any) -> bool:
    """比较resource_name字段，支持单个资源名称或资源名称数组"""
    try:
        # 解析resource_name字段中的JSON字符串
        def parse_resource_name(value: Any) -> Any:
            # 如果是形如 {"resource_name": "..."} 或 {"resource_name": ["...", "..."]} 的结构，提取其内部值
            if isinstance(value, dict) and 'resource_name' in value:
                return _parse_possible_json_string(value['resource_name'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_resource_name(predict_json)
        label_data = parse_resource_name(label_json)
        
        # 如果都是列表，比较排序后的结果（忽略顺序）
        if isinstance(predict_data, list) and isinstance(label_data, list):
            return sorted(predict_data) == sorted(label_data)
        
        # 如果都是字符串或其他相同类型，直接比较
        if type(predict_data) == type(label_data):
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_resource_detail(predict_json: Any, label_json: Any) -> bool:
    """比较resource_detail字段，处理null值的特殊逻辑"""
    try:
        # 解析resource_detail字段中的JSON字符串（鲁棒解析）
        def parse_resource_detail(value: Any) -> Any:
            # 如果是形如 {"resource_detail": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 'resource_detail' in value:
                return _parse_possible_json_string(value['resource_detail'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_resource_detail(predict_json)
        label_data = parse_resource_detail(label_json)
        
        # 如果都是字典，进行智能比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 标准化处理：将缺失的字段视为null，将显式的null和缺失字段统一处理
            def normalize_dict(d):
                """标准化字典，将所有可能的字段都补全，缺失的设为None"""
                all_keys = set(predict_data.keys()) | set(label_data.keys())
                return {k: d.get(k) for k in all_keys}
            
            predict_normalized = normalize_dict(predict_data)
            label_normalized = normalize_dict(label_data)
            
            # 比较标准化后的字典（缺失与null等价）
            return predict_normalized == label_normalized
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def evaluate_field_accuracy(predictions_file: str) -> Dict[str, Dict[str, Any]]:
    """评估每个字段的准确率"""
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'correct_examples': [], 'error_examples': []})
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 提取instruction，predict和label
                prompt = data.get('prompt', '')
                predict_raw = data.get('predict', '')
                label_raw = data.get('label', '')
                
                # 从prompt中提取instruction和input
                instruction_match = re.search(r'instruction:\s*(.*?)(?=<\|im_end\|>|$)', prompt, re.DOTALL)
                if not instruction_match:
                    continue
                
                instruction = instruction_match.group(1).strip()
                
                # 提取input
                input_match = re.search(r'input:\s*(.*?)(?=\s*instruction:)', prompt, re.DOTALL)
                input_text = input_match.group(1).strip() if input_match else ""
                
                field_name = extract_field_from_instruction(instruction)
                
                # 调试：检查特定字段的识别
                if '请根据以上用户的查询' in instruction or '请根据query和工具信息' in instruction:
                    print(f"调试 - Line {line_num}: 识别到相关指令，字段名: {field_name}")
                    print(f"调试 - 指令内容: {instruction[:100]}...")
                
                # 跳过customer_name字段的统计
                if field_name == 'customer_name':
                    continue
                
                # 解析预测和标签的JSON
                predict_json = extract_json_from_response(predict_raw)
                label_json = extract_json_from_response(label_raw)
                
                # 对于 final_answer 和 user_intent_analysis 字段，如果JSON解析失败，使用原始文本
                if field_name in ['final_answer', 'user_intent_analysis']:
                    if predict_json is None:
                        predict_json = predict_raw
                    if label_json is None:
                        label_json = label_raw
                elif predict_json is None or label_json is None:
                    continue
                
                # 使用智能比较
                is_correct = smart_compare_json(predict_json, label_json, field_name)
                
                field_stats[field_name]['total'] += 1
                if is_correct:
                    field_stats[field_name]['correct'] += 1
                
                # 保存示例用于分析 - 分别保存正确和错误的示例
                example_data = {
                    'line': line_num,
                    'instruction': instruction,
                    'input': input_text,
                    'predict': predict_json,
                    'label': label_json,
                    'correct': is_correct
                }
                
                if is_correct:
                    # 正确的示例只保留前3个
                    if len(field_stats[field_name]['correct_examples']) < 3:
                        field_stats[field_name]['correct_examples'].append(example_data)
                else:
                    # 错误的示例保留所有
                    field_stats[field_name]['error_examples'].append(example_data)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    # 调试信息：显示字段统计
    print("Field stats keys:", list(field_stats.keys()))
    for field_name, stats in field_stats.items():
        if field_name in ['final_answer', 'user_intent_analysis']:
            print(f"{field_name}: {stats}")
    
    # 调试：检查为什么没有 final_answer 和 user_intent_analysis
    print("\n=== 调试信息 ===")
    print("检查 final_answer 和 user_intent_analysis 字段识别问题...")
    
    return dict(field_stats)


def generate_report(field_stats: Dict[str, Dict[str, Any]]) -> str:
    """生成评估报告"""
    report = ["#### 按字段分类的准确率评估报告"]
    report.append("")
    
    # 添加新字段分类说明
    report.append("##### 字段分类说明")
    report.append("")
    report.append("**新增字段（要求完全准确率）**:")
    report.append("- `service_name`: 服务类型识别（景区门票/酒店住宿/索道服务等）")
    report.append("- `node_name`: 节点名称识别（产品节点和价格条件）")
    report.append("- `re_rank`: API重排序（根据功能匹配度重新排序候选API）")
    report.append("- `final_answer`: 最终回答生成（根据用户查询和工具调用结果生成回答）")
    report.append("- `user_intent_analysis`: 用户意图分析（根据query和工具信息分析用户意图）")
    report.append("- `price_policy_calculation`: 价格政策计算（根据价格政策和OCR文本计算资源价格）")
    report.append("")
    report.append("**原有字段**:")
    report.append("- 其他字段保持原有的智能比较逻辑")
    report.append("")
    
    # 计算总体统计
    total_correct = sum(stats['correct'] for stats in field_stats.values())
    total_samples = sum(stats['total'] for stats in field_stats.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    report.append("##### 总体统计")
    report.append("")
    report.append(f"- **总样本数**: {total_samples}")
    report.append(f"- **总正确数**: {total_correct}")
    report.append(f"- **总体准确率**: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    report.append("")
    
    # 按准确率排序
    sorted_fields = sorted(field_stats.items(), 
                          key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total'] > 0 else 0,
                          reverse=True)
    
    report.append("##### 各字段详细统计")
    report.append("")
    
    for field_name, stats in sorted_fields:
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        report.append(f"###### {field_name}")
        report.append("")
        report.append(f"- **样本数**: {stats['total']}")
        report.append(f"- **正确数**: {stats['correct']}")
        report.append(f"- **准确率**: {accuracy:.4f} ({accuracy*100:.2f}%)")
        report.append("")
        
        # 显示示例
        # 显示正确示例（限制数量）
        if stats['correct_examples']:
            report.append("**正确示例**:")
            report.append("")
            for i, example in enumerate(stats['correct_examples'][:3], 1):
                report.append(f"{i}. ✅ **Line {example['line']}**")
                report.append(f"   - 预测: `{json.dumps(example['predict'], ensure_ascii=False)}`")
                report.append(f"   - 标签: `{json.dumps(example['label'], ensure_ascii=False)}`")
                report.append("")
        
        # 为新字段添加特殊说明
        if field_name in ['service_name', 'node_name', 're_rank', 'final_answer', 'user_intent_analysis', 'price_policy_calculation']:
            report.append("**⚠️ 重要说明**: 此字段要求完全准确率，任何细微差异都将被视为错误")
            report.append("")
        
        # 显示所有错误示例
        if stats['error_examples']:
            report.append("**错误示例（全部）**:")
            report.append("")
            for i, example in enumerate(stats['error_examples'], 1):
                report.append(f"{i}. ❌ **Line {example['line']}**")
                report.append("")
                report.append(f"   **完整指令**:")
                report.append(f"   ```")
                report.append(f"   {example['instruction']}")
                report.append(f"   ```")
                report.append("")
                if example['input']:
                    report.append(f"   **完整输入**:")
                    report.append(f"   ```")
                    report.append(f"   {example['input']}")
                    report.append(f"   ```")
                    report.append("")
                report.append(f"   - **预测**: `{json.dumps(example['predict'], ensure_ascii=False)}`")
                report.append(f"   - **标签**: `{json.dumps(example['label'], ensure_ascii=False)}`")
                report.append("")
        
        report.append("---")
        report.append("")
    
    return "\n".join(report)


def compare_price_policy_calculation(predict_json: Any, label_json: Any) -> bool:
    """比较价格政策计算字段，需要完全准确率"""
    try:
        # 解析价格政策计算字段中的JSON字符串
        def parse_price_policy(value: Any) -> Any:
            # 如果是形如 {"price_policy": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 'price_policy' in value:
                return _parse_possible_json_string(value['price_policy'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_price_policy(predict_json)
        label_data = parse_price_policy(label_json)
        
        # 如果都是字典，进行完全比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 对于价格政策计算，需要完全匹配，包括所有字段和数值
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_user_intent_analysis(predict_json: Any, label_json: Any) -> bool:
    """比较用户意图分析字段，需要完全准确率"""
    try:
        # 如果输入是字符串，直接比较文本内容
        if isinstance(predict_json, str) and isinstance(label_json, str):
            # 移除<think>标签和多余空格后比较
            predict_clean = re.sub(r'<think>.*?</think>', '', predict_json, flags=re.DOTALL).strip()
            label_clean = re.sub(r'<think>.*?</think>', '', label_json, flags=re.DOTALL).strip()
            return predict_clean == label_clean
        
        # 解析用户意图分析字段中的JSON字符串
        def parse_user_intent(value: Any) -> Any:
            # 如果是形如 {"user_intent": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 'user_intent' in value:
                return _parse_possible_json_string(value['user_intent'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_user_intent(predict_json)
        label_data = parse_user_intent(label_json)
        
        # 如果都是字典，进行完全比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 对于用户意图分析，需要完全匹配，包括所有参数
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_re_rank(predict_json: Any, label_json: Any) -> bool:
    """比较re-rank字段，需要完全准确率"""
    try:
        # 解析re-rank字段中的JSON字符串
        def parse_re_rank(value: Any) -> Any:
            # 如果是形如 {"re_rank": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 're_rank' in value:
                return _parse_possible_json_string(value['re_rank'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_re_rank(predict_json)
        label_data = parse_re_rank(label_json)
        
        # 如果都是字典，进行完全比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 对于re-rank，需要完全匹配，包括所有API排序
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_node_name(predict_json: Any, label_json: Any) -> bool:
    """比较node_name字段，需要完全准确率"""
    try:
        # 解析node_name字段中的JSON字符串
        def parse_node_name(value: Any) -> Any:
            # 如果是形如 {"node_name": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 'node_name' in value:
                return _parse_possible_json_string(value['node_name'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_node_name(predict_json)
        label_data = parse_node_name(label_json)
        
        # 如果都是字典，进行完全比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 对于node_name，需要完全匹配，包括所有节点信息
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_service_name(predict_json: Any, label_json: Any) -> bool:
    """比较service_name字段，需要完全准确率"""
    try:
        # 解析service_name字段中的JSON字符串
        def parse_service_name(value: Any) -> Any:
            # 如果是形如 {"service_name": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 'service_name' in value:
                return _parse_possible_json_string(value['service_name'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_service_name(predict_json)
        label_data = parse_service_name(label_json)
        
        # 如果都是字典，进行完全比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 对于service_name，需要完全匹配，包括服务类型、项目名称和主要产品
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def compare_final_answer(predict_json: Any, label_json: Any) -> bool:
    """比较final_answer字段，需要完全准确率"""
    try:
        # 如果输入是字符串，直接比较文本内容
        if isinstance(predict_json, str) and isinstance(label_json, str):
            # 移除<think>标签和多余空格后比较
            predict_clean = re.sub(r'<think>.*?</think>', '', predict_json, flags=re.DOTALL).strip()
            label_clean = re.sub(r'<think>.*?</think>', '', label_json, flags=re.DOTALL).strip()
            return predict_clean == label_clean
        
        # 解析final_answer字段中的JSON字符串
        def parse_final_answer(value: Any) -> Any:
            # 如果是形如 {"final_answer": "{...}"} 的结构，提取其内部JSON
            if isinstance(value, dict) and 'final_answer' in value:
                return _parse_possible_json_string(value['final_answer'])
            # 如果直接就是一个JSON字符串或对象，也尝试解析
            return _parse_possible_json_string(value)

        predict_data = parse_final_answer(predict_json)
        label_data = parse_final_answer(label_json)
        
        # 如果都是字典，进行完全比较
        if isinstance(predict_data, dict) and isinstance(label_data, dict):
            # 对于final_answer，需要完全匹配，包括所有回答内容
            return predict_data == label_data
        
        # 如果结构不同，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)
        
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，使用标准比较
        return normalize_value(predict_json) == normalize_value(label_json)


def main():
    """主函数"""
    predictions_file = "/home/ziqiang/LLaMA-Factory/Qwen3-8B/eval_results/9_14/generated_predictions.jsonl"
    
    print("开始分析预测结果...")
    field_stats = evaluate_field_accuracy(predictions_file)
    
    print("生成评估报告...")
    report = generate_report(field_stats)
    
    # 计算总体统计用于控制台输出
    total_correct = sum(stats['correct'] for stats in field_stats.values())
    total_samples = sum(stats['total'] for stats in field_stats.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # 简要输出到控制台
    print(f"总体准确率: {total_correct}/{total_samples} = {overall_accuracy*100:.2f}%")
    
    # 保存到文件
    output_file = "/home/ziqiang/LLaMA-Factory/Qwen3-8B/eval_results/9_14/field_accuracy_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {output_file}")
    
    # 生成JSON格式的详细结果
    json_output = {}
    for field_name, stats in field_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        json_output[field_name] = {
            'total_samples': stats['total'],
            'correct_samples': stats['correct'],
            'error_samples': len(stats['error_examples']),
            'accuracy': accuracy,
            'correct_examples': stats['correct_examples'],
            'error_examples': stats['error_examples']
        }
    
    json_output_file = "/home/ziqiang/LLaMA-Factory/Qwen3-8B/eval_results/9_14/field_accuracy_detailed.json"
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存到: {json_output_file}")


if __name__ == "__main__":
    main()
