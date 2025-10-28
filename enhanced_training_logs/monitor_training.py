#!/usr/bin/env python3
'''
增强训练监控脚本
实时监控训练过程中的以下内容：

1. Label和Token分析
   - 数据切分情况
   - Token变化追踪
   - 中文Token解码

2. 预测Token监控
   - 模型预测输出变化
   - 预测准确率统计
   - 预测文本对比

3. 训练过程监控
   - Loss变化
   - 学习率调整
   - 训练进度
'''

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

def setup_loggers(log_files):
    '''设置日志记录器'''
    loggers = {}
    
    for log_type, log_file in log_files.items():
        logger = logging.getLogger(f"monitor_{log_type}")
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        loggers[log_type] = logger
        
    return loggers

def monitor_training_logs(loggers, log_files):
    '''监控训练日志'''
    
    main_logger = loggers["main"]
    main_logger.info("🔍 开始监控训练过程")
    
    # 监控训练日志文件
    trainer_log = None
    for log_file in log_files.values():
        if "main_training" in log_file:
            trainer_log = log_file
            break
    
    if trainer_log:
        main_logger.info(f"📝 监控训练日志: {trainer_log}")
        
        # 监控文件变化
        last_size = 0
        while True:
            try:
                if os.path.exists(trainer_log):
                    current_size = os.path.getsize(trainer_log)
                    if current_size > last_size:
                        # 读取新增内容
                        with open(trainer_log, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            
                        # 记录新内容
                        for line in new_content.strip().split('\n'):
                            if line.strip():
                                main_logger.info(f"📊 训练日志: {line}")
                                
                        last_size = current_size
                        
                time.sleep(1)  # 每秒检查一次
                
            except KeyboardInterrupt:
                main_logger.info("🛑 监控已停止")
                break
            except Exception as e:
                main_logger.error(f"❌ 监控错误: {e}")
                time.sleep(5)
    else:
        main_logger.warning("⚠️ 未找到训练日志文件")

def main():
    '''主函数'''
    
    log_files = {"training": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/training_monitor_20251027_182728.log", "predictions": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251027_182728.log", "labels": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20251027_182728.log", "alignment": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251027_182728.log", "main": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251027_182728.log"}
    
    # 设置日志记录器
    loggers = setup_loggers(log_files)
    
    # 开始监控
    monitor_training_logs(loggers, log_files)

if __name__ == "__main__":
    main()
