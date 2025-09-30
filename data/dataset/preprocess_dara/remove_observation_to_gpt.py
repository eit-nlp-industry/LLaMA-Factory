#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除JSON文件中指定数据条目的observation到gpt部分
"""

import json
import argparse
import sys
from typing import List, Dict, Any


def remove_observation_to_gpt(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    删除conversations中从observation到gpt的部分
    
    Args:
        conversations: 对话列表
        
    Returns:
        处理后的对话列表
    """
    result = []
    skip_until_gpt = False
    
    for item in conversations:
        if item.get("from") == "observation":
            # 遇到observation，开始跳过
            skip_until_gpt = True
            continue
        elif item.get("from") == "gpt" and skip_until_gpt:
            # 遇到gpt且正在跳过，停止跳过
            skip_until_gpt = False
            continue
        elif skip_until_gpt:
            # 正在跳过中，继续跳过
            continue
        else:
            # 正常添加
            result.append(item)
    
    return result


def process_json_file(input_file: str, output_file: str = None, data_index: int = None) -> None:
    """
    处理JSON文件，删除指定数据条目的observation到gpt部分
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
        data_index: 要处理的数据条目索引，如果为None则处理所有条目
    """
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("错误：JSON文件应该包含一个数组")
            return
        
        # 处理数据
        if data_index is not None:
            # 处理指定索引的数据
            if 0 <= data_index < len(data):
                if "conversations" in data[data_index]:
                    original_count = len(data[data_index]["conversations"])
                    data[data_index]["conversations"] = remove_observation_to_gpt(data[data_index]["conversations"])
                    new_count = len(data[data_index]["conversations"])
                    print(f"数据条目 {data_index}: 删除了 {original_count - new_count} 个对话项")
                else:
                    print(f"数据条目 {data_index} 中没有conversations字段")
            else:
                print(f"错误：数据索引 {data_index} 超出范围 (0-{len(data)-1})")
                return
        else:
            # 处理所有数据
            total_removed = 0
            for i, item in enumerate(data):
                if "conversations" in item:
                    original_count = len(item["conversations"])
                    item["conversations"] = remove_observation_to_gpt(item["conversations"])
                    new_count = len(item["conversations"])
                    removed = original_count - new_count
                    if removed > 0:
                        print(f"数据条目 {i}: 删除了 {removed} 个对话项")
                        total_removed += removed
            
            print(f"总共删除了 {total_removed} 个对话项")
        
        # 保存文件
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，结果已保存到: {output_path}")
        
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
    except json.JSONDecodeError as e:
        print(f"错误：JSON文件格式错误 - {e}")
    except Exception as e:
        print(f"错误：{e}")


def main():
    parser = argparse.ArgumentParser(description="删除JSON文件中指定数据条目的observation到gpt部分")
    parser.add_argument("input_file", help="输入JSON文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（可选，默认覆盖原文件）")
    parser.add_argument("-i", "--index", type=int, help="要处理的数据条目索引（可选，默认处理所有条目）")
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output, args.index)


if __name__ == "__main__":
    main()
