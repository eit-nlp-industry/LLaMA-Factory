#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集集成脚本：将身份证数据集成到OCR文本训练数据中
同时改进身份证数据的instruction格式
"""

import json
import os
from typing import List, Dict, Any


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {file_path}，包含 {len(data)} 条记录")
        return data
    except Exception as e:
        print(f"加载文件失败 {file_path}: {e}")
        return []


def improve_idcard_instruction(original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    改进身份证数据的instruction格式，参考OCR文本数据的详细写法
    """
    improved_data = []
    
    # 新的详细instruction模板
    detailed_instruction = """请从OCR文本中抽取旅行订单中的游客身份证信息。

提取字段及说明：
name (姓名): 游客的真实姓名
idcard (身份证号): 18位身份证号码，支持末位为X的格式
gender (性别): 根据身份证号倒数第二位数字判断（奇数为男，偶数为女）
phone (电话号码): 游客联系电话，如果OCR文本中没有则为null

严格按照以下JSON格式输出：
[
  {
    "name": "姓名",
    "idcard": "身份证号",
    "gender": "男/女",
    "phone": "电话号码或null"
  }
]

注意事项：
1. 确保身份证号码格式正确（18位数字，末位可为X）
2. 性别根据身份证号自动判断，不依赖OCR文本中的性别信息
3. 如果姓名缺失或无法识别，name字段设为"无"
4. 电话号码如果在OCR文本中不存在，设置为null
5. 输出必须是有效的JSON数组格式"""

    for item in original_data:
        improved_item = item.copy()
        improved_item['instruction'] = detailed_instruction
        improved_data.append(improved_item)
    
    print(f"改进了 {len(improved_data)} 条身份证数据的instruction格式")
    return improved_data


def integrate_datasets(ocr_text_data: List[Dict[str, Any]], 
                      idcard_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    集成两个数据集
    """
    print("正在集成数据集...")
    
    # 合并数据集
    integrated_data = ocr_text_data + idcard_data
    
    print(f"集成完成：")
    print(f"- OCR文本数据：{len(ocr_text_data)} 条")
    print(f"- 身份证数据：{len(idcard_data)} 条")
    print(f"- 总计：{len(integrated_data)} 条")
    
    return integrated_data


def save_integrated_dataset(data: List[Dict[str, Any]], output_path: str):
    """
    保存集成后的数据集
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"集成数据集已保存到: {output_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")


def validate_dataset(data: List[Dict[str, Any]]) -> bool:
    """
    验证数据集格式正确性
    """
    print("正在验证数据集格式...")
    
    required_fields = ['instruction', 'input', 'output']
    issues = []
    
    for i, item in enumerate(data):
        # 检查必要字段
        for field in required_fields:
            if field not in item:
                issues.append(f"记录 {i}: 缺少字段 '{field}'")
        
        # 检查instruction是否为空
        if 'instruction' in item and not item['instruction'].strip():
            issues.append(f"记录 {i}: instruction字段为空")
        
        # 检查output是否为有效JSON
        if 'output' in item:
            try:
                json.loads(item['output'])
            except json.JSONDecodeError:
                issues.append(f"记录 {i}: output字段不是有效的JSON格式")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  - ... 还有 {len(issues) - 10} 个问题")
        return False
    else:
        print("数据集格式验证通过！")
        return True


def main():
    """
    主函数
    """
    print("开始数据集集成任务...")
    
    # 文件路径
    ocr_text_file = "/home/ziqiang/LLaMA-Factory/data/ocr_text_orders_08_18.json"
    idcard_file = "/home/ziqiang/LLaMA-Factory/data/ocr_idcards_orders.json"
    output_file = "/home/ziqiang/LLaMA-Factory/data/text_idcards_8_18.json"
    
    # 加载数据
    print("\n1. 加载原始数据集...")
    ocr_text_data = load_json_data(ocr_text_file)
    idcard_data = load_json_data(idcard_file)
    
    if not ocr_text_data or not idcard_data:
        print("数据加载失败，终止执行")
        return
    
    # 改进身份证数据的instruction
    print("\n2. 改进身份证数据的instruction格式...")
    improved_idcard_data = improve_idcard_instruction(idcard_data)
    
    # 集成数据集
    print("\n3. 集成数据集...")
    integrated_data = integrate_datasets(ocr_text_data, improved_idcard_data)
    
    # 验证数据集
    print("\n4. 验证数据集格式...")
    if not validate_dataset(integrated_data):
        print("数据集验证失败，请检查数据格式")
        return
    
    # 保存集成后的数据集
    print("\n5. 保存集成数据集...")
    save_integrated_dataset(integrated_data, output_file)
    
    print(f"\n✅ 数据集集成完成！")
    print(f"集成后的数据集保存在: {output_file}")
    print(f"总记录数: {len(integrated_data)}")
    
    # 显示样本统计
    ocr_count = len(ocr_text_data)
    idcard_count = len(improved_idcard_data)
    print(f"\n数据构成:")
    print(f"- OCR文本数据: {ocr_count} 条 ({ocr_count/len(integrated_data)*100:.1f}%)")
    print(f"- 身份证数据: {idcard_count} 条 ({idcard_count/len(integrated_data)*100:.1f}%)")


if __name__ == "__main__":
    main()
