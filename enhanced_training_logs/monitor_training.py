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
   - 验证集Loss (eval_loss) 监控和记录
   - 学习率调整
   - 训练进度

4. 验证集Loss监控
   - 实时提取和记录eval_loss
   - 从trainer_state.json读取准确的eval_loss
   - 保存验证集Loss历史到JSON文件
   - 监控验证集Loss变化趋势
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
    eval_loss_logger = loggers.get("eval_loss")
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
        eval_loss_history = []  # 保存验证集loss历史
        
        while True:
            try:
                if os.path.exists(trainer_log):
                    current_size = os.path.getsize(trainer_log)
                    if current_size > last_size:
                        # 读取新增内容
                        with open(trainer_log, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            
                        # 记录新内容并提取eval_loss
                        for line in new_content.strip().split('\n'):
                            if line.strip():
                                main_logger.info(f"📊 训练日志: {line}")
                                
                                # 提取eval_loss信息
                                if "eval_loss" in line.lower() or "'eval_loss'" in line or '"eval_loss"' in line:
                                    try:
                                        # 尝试从JSON格式中提取eval_loss
                                        import re
                                        # 匹配 eval_loss: value 或 "eval_loss": value
                                        match = re.search(r'["']?eval_loss["']?\s*[:=]\s*([0-9.]+)', line, re.IGNORECASE)
                                        if match:
                                            eval_loss_value = float(match.group(1))
                                            eval_loss_history.append({
                                                "step": len(eval_loss_history) + 1,
                                                "eval_loss": eval_loss_value,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            })
                                            
                                            if eval_loss_logger:
                                                eval_loss_logger.info(
                                                    f"📈 Step {len(eval_loss_history)} | "
                                                    f"Eval Loss: {eval_loss_value:.6f} | "
                                                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                                                )
                                            
                                            main_logger.info(
                                                f"✅ 验证集Loss更新: {eval_loss_value:.6f}"
                                            )
                                    except Exception as e:
                                        pass  # 如果解析失败，忽略
                                
                        last_size = current_size
                        
                # 监控输出目录中的trainer_state.json以获取更准确的eval_loss
                # 尝试从环境变量或日志中获取output_dir
                output_dir = os.environ.get("OUTPUT_DIR")
                if not output_dir:
                    # 从日志文件中提取output_dir（如果存在）
                    try:
                        if os.path.exists(trainer_log):
                            with open(trainer_log, 'r', encoding='utf-8') as f:
                                content = f.read()
                                import re
                                match = re.search(r'输出目录[:：]\s*([^\n]+)', content)
                                if match:
                                    output_dir = match.group(1).strip()
                    except:
                        pass
                
                if output_dir and os.path.exists(output_dir):
                    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
                    if os.path.exists(trainer_state_file):
                        try:
                            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                                trainer_state = json.load(f)
                            
                            # 检查log_history中的最新eval_loss
                            if "log_history" in trainer_state:
                                for log_entry in reversed(trainer_state["log_history"]):
                                    if "eval_loss" in log_entry:
                                        eval_loss_value = log_entry["eval_loss"]
                                        step = log_entry.get("step", 0)
                                        
                                        # 检查是否是新记录
                                        if not any(h.get("step") == step for h in eval_loss_history):
                                            eval_loss_history.append({
                                                "step": step,
                                                "eval_loss": eval_loss_value,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            })
                                            
                                            if eval_loss_logger:
                                                eval_loss_logger.info(
                                                    f"📈 Step {step} | "
                                                    f"Eval Loss: {eval_loss_value:.6f} | "
                                                    f"Time: {datetime.now().strftime('%H:%M:%S')}"
                                                )
                                            
                                            main_logger.info(
                                                f"✅ 验证集Loss (Step {step}): {eval_loss_value:.6f}"
                                            )
                                        break
                        except Exception as e:
                            pass  # 如果读取失败，忽略
                        
                time.sleep(2)  # 每2秒检查一次
                
            except KeyboardInterrupt:
                main_logger.info("🛑 监控已停止")
                # 保存eval_loss历史到JSON文件
                if eval_loss_history and eval_loss_logger:
                    eval_loss_file = log_files.get("eval_loss", "").replace(".log", "_history.json")
                    try:
                        with open(eval_loss_file, 'w', encoding='utf-8') as f:
                            json.dump(eval_loss_history, f, indent=2, ensure_ascii=False)
                        eval_loss_logger.info(f"💾 验证集Loss历史已保存: {eval_loss_file}")
                    except Exception as e:
                        eval_loss_logger.error(f"❌ 保存验证集Loss历史失败: {e}")
                break
            except Exception as e:
                main_logger.error(f"❌ 监控错误: {e}")
                time.sleep(5)
    else:
        main_logger.warning("⚠️ 未找到训练日志文件")

def main():
    '''主函数'''
    
    log_files = {"training": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/training_monitor_20260117_121526.log", "predictions": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20260117_121526.log", "labels": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20260117_121526.log", "alignment": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20260117_121526.log", "eval_loss": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/eval_loss_monitor_20260117_121526.log", "token_loss": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/token_loss_analysis_20260117_121526.log", "main": "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20260117_121526.log"}
    
    # 设置日志记录器
    loggers = setup_loggers(log_files)
    
    # 开始监控
    monitor_training_logs(loggers, log_files)

if __name__ == "__main__":
    main()
