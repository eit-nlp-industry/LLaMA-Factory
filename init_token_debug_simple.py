#!/usr/bin/env python3
"""
简单的token调试日志初始化脚本
在训练开始前运行此脚本来配置日志
"""

import os
import sys
from datetime import datetime

def init_simple_logger():
    """初始化简单的日志配置"""
    try:
        from loguru import logger
        
        # 移除默认handler
        logger.remove()
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"token_debug_{timestamp}.log"
        
        # 配置文件输出
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation="50 MB",
            retention="3 days",
            enqueue=True
        )
        
        # 同时输出到控制台
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
            level="INFO",
            colorize=True
        )
        
        logger.info(f"Token调试日志已配置，保存到: {os.path.abspath(log_file)}")
        return log_file
        
    except ImportError:
        print("❌ 未安装loguru，将使用print输出")
        return None

if __name__ == "__main__":
    log_file = init_simple_logger()
    if log_file:
        print(f"✅ 日志文件: {log_file}")
    else:
        print("❌ 日志配置失败")
