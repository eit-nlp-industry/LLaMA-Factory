#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试标签平衡功能的脚本
"""

import json
import os
from quick_balance import quick_balance


def test_label_balance():
    """测试标签平衡功能"""
    input_file = "ocr_text_orders_20250101_20250715.json"
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    print("开始测试标签平衡功能...")
    print("=" * 50)
    
    try:
        # 测试不同的max_samples参数
        for max_samples in [30, 50, 100]:
            print(f"\n测试 max_samples = {max_samples}")
            print("-" * 30)
            
            output_file = f"test_balanced_{max_samples}.json"
            result = quick_balance(input_file, output_file, max_samples)
            
            print(f"结果: {len(result)} 条记录")
            
            # 验证输出文件
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024  # KB
                print(f"输出文件大小: {file_size:.2f} KB")
            else:
                print("警告: 输出文件未生成")
        
        print("\n所有测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_label_balance()





