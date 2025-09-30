#!/usr/bin/env python3
"""
启用token调试日志的脚本
在训练前运行此脚本设置日志环境
"""

import os
import sys
from datetime import datetime

def setup_token_debug_env():
    """设置token调试环境变量"""
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"/home/ziqiang/LLaMA-Factory/token_debug_{timestamp}.log"
    
    # 设置环境变量
    os.environ["TOKEN_DEBUG_LOG_FILE"] = log_file
    os.environ["TOKEN_DEBUG_ENABLED"] = "1"
    
    print(f"✅ Token调试已启用")
    print(f"📝 日志文件: {log_file}")
    print(f"🔧 环境变量已设置:")
    print(f"   TOKEN_DEBUG_LOG_FILE={log_file}")
    print(f"   TOKEN_DEBUG_ENABLED=1")
    
    return log_file

if __name__ == "__main__":
    setup_token_debug_env()
