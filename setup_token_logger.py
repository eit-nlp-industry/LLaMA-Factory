#!/usr/bin/env python3
"""
Token调试日志配置
统一配置loguru将所有token调试日志保存到指定文件
"""

from loguru import logger
import sys
import os
from datetime import datetime

def setup_token_debug_logger(log_file_path=None):
    """
    配置token调试专用的logger
    
    Args:
        log_file_path: 日志文件路径，如果为None则自动生成
    """
    # 移除默认的控制台handler
    logger.remove()
    
    # 如果没有指定日志文件路径，自动生成
    if log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"token_debug_{timestamp}.log"
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "."
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置文件输出 - 包含所有级别的日志
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",  # 文件大小达到100MB时轮转
        retention="7 days",  # 保留7天的日志
        compression="zip",   # 压缩旧日志
        enqueue=True,       # 异步写入，提高性能
        backtrace=True,     # 显示完整的错误堆栈
        diagnose=True       # 显示变量值
    )
    
    # 同时输出到控制台 - 只显示INFO及以上级别
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    logger.info(f"Token调试日志已配置，文件保存到: {os.path.abspath(log_file_path)}")
    return log_file_path

def get_token_debug_logger():
    """
    获取配置好的token调试logger
    """
    return logger

if __name__ == "__main__":
    # 测试日志配置
    log_file = setup_token_debug_logger("token_debug_test.log")
    
    test_logger = get_token_debug_logger()
    test_logger.info("[TOKEN_DEBUG] 测试日志配置")
    test_logger.info("[TEMPLATE_DEBUG] 模板调试日志测试")
    test_logger.info("[INFER_SEQLEN] 截断策略日志测试")
    
    print(f"测试完成，日志文件: {log_file}")
