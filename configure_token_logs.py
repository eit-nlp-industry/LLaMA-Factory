#!/usr/bin/env python3
"""
Token调试日志配置脚本
简化版本，直接配置loguru将日志保存到文件
"""

import os
import sys
from datetime import datetime

def configure_token_logger():
    """配置token调试日志到文件"""
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

# 在模块导入时自动配置
_log_file = configure_token_logger()

def get_logger():
    """获取配置好的logger"""
    try:
        from loguru import logger
        return logger
    except ImportError:
        # 如果没有loguru，创建一个简单的logger替代
        class SimpleLogger:
            def info(self, msg):
                print(f"INFO | {msg}")
            def debug(self, msg):
                print(f"DEBUG | {msg}")
            def warning(self, msg):
                print(f"WARNING | {msg}")
            def error(self, msg):
                print(f"ERROR | {msg}")
        return SimpleLogger()

if __name__ == "__main__":
    logger = get_logger()
    logger.info("[TOKEN_DEBUG] 测试日志")
    logger.info("[TEMPLATE_DEBUG] 测试模板日志")
    logger.info("[INFER_SEQLEN] 测试截断日志")
    
    if _log_file:
        print(f"日志文件: {_log_file}")
    else:
        print("使用控制台输出")
