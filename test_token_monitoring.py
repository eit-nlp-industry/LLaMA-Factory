#!/usr/bin/env python3
"""
测试Token监控功能
"""

import os
import sys
import subprocess
from datetime import datetime

def test_token_monitoring():
    """测试Token监控功能"""
    
    print("🧪 测试Token监控功能")
    print("=" * 60)
    
    # 运行增强训练脚本
    print("🚀 运行增强训练脚本...")
    
    try:
        result = subprocess.run([
            "python", "/home/ziqiang/LLaMA-Factory/create_enhanced_training.py"
        ], cwd="/home/ziqiang/LLaMA-Factory", capture_output=False, text=True)
        
        print("✅ 测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_token_monitoring()

