#!/usr/bin/env python3
"""
Token调试日志初始化脚本
在训练开始前运行此脚本来配置统一的日志文件
"""

import os
import sys
from datetime import datetime

def init_token_debug_logging(output_dir=None):
    """
    初始化token调试日志
    
    Args:
        output_dir: 输出目录，如果为None则使用当前目录
    """
    if output_dir is None:
        output_dir = "."
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"token_debug_{timestamp}.log")
    
    # 导入并配置logger
    try:
        from setup_token_logger import setup_token_debug_logger
        actual_log_file = setup_token_debug_logger(log_file)
        
        print(f"✅ Token调试日志已初始化")
        print(f"📝 日志文件: {actual_log_file}")
        print(f"🔍 查看日志: tail -f {actual_log_file}")
        print(f"🔎 过滤日志: grep 'TOKEN_DEBUG' {actual_log_file}")
        
        return actual_log_file
        
    except ImportError as e:
        print(f"❌ 无法导入日志配置: {e}")
        print("请确保setup_token_logger.py文件存在且可访问")
        return None

if __name__ == "__main__":
    # 从命令行参数获取输出目录
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("🚀 初始化Token调试日志系统...")
    log_file = init_token_debug_logging(output_dir)
    
    if log_file:
        print("\n📋 使用方法:")
        print("1. 运行训练命令")
        print("2. 日志会自动保存到指定文件")
        print("3. 使用以下命令查看日志:")
        print(f"   tail -f {log_file}")
        print(f"   grep 'TOKEN_DEBUG' {log_file}")
        print(f"   grep 'INFER_SEQLEN' {log_file}")
        print(f"   grep 'TEMPLATE_DEBUG' {log_file}")
    else:
        print("❌ 初始化失败")
        sys.exit(1)
