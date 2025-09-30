#!/usr/bin/env python3
"""
简化的增强训练脚本
直接基于LLaMA-Factory的原始训练流程，添加监控功能
"""

import os
import sys
import subprocess
from datetime import datetime

def create_enhanced_training_command():
    """创建增强的训练命令"""
    
    # 基础训练命令
    base_cmd = [
        "CUDA_VISIBLE_DEVICES=6",
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--do_train", "True",
        "--model_name_or_path", "/data/models/Qwen3-8B",
        "--preprocessing_num_workers", "1",
        "--finetuning_type", "lora",
        "--template", "qwen3",
        "--flash_attn", "auto",
        "--dataset_dir", "data",
        "--dataset", "data_demo",
        "--cutoff_len", "8192",
        "--learning_rate", "1.0e-5",
        "--num_train_epochs", "50.0",
        "--max_samples", "100000",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--lr_scheduler_type", "cosine",
        "--max_grad_norm", "1.0",
        "--logging_steps", "1",
        "--save_steps", "10",
        "--warmup_steps", "0",
        "--packing", "False",
        "--enable_thinking", "False",
        "--overwrite_cache", "True",
        "--bf16", "True",
        "--plot_loss", "True",
        "--trust_remote_code", "True",
        "--ddp_timeout", "180000000",
        "--include_num_input_tokens_seen", "True",
        "--optim", "adamw_torch",
        "--lora_rank", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0",
        "--lora_target", "all",
        "--gradient_checkpointing", "True"
    ]
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_{timestamp}"
    base_cmd.extend(["--output_dir", output_dir])
    
    return " ".join(base_cmd), output_dir

def setup_enhanced_logging(output_dir):
    """设置增强的日志记录"""
    
    # 创建日志目录
    log_dir = os.path.join(output_dir, "enhanced_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files = {
        "training": os.path.join(log_dir, f"training_monitor_{timestamp}.log"),
        "predictions": os.path.join(log_dir, f"prediction_monitor_{timestamp}.log"),
        "labels": os.path.join(log_dir, f"label_analysis_{timestamp}.log"),
        "alignment": os.path.join(log_dir, f"alignment_analysis_{timestamp}.log")
    }
    
    return log_files

def create_monitoring_script(output_dir, log_files):
    """创建监控脚本"""
    
    monitoring_script = f"""#!/usr/bin/env python3
'''
训练监控脚本
实时监控训练过程中的label、predict和对齐情况
'''

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加LLaMA-Factory路径
sys.path.insert(0, "/home/ziqiang/LLaMA-Factory/src")

from llamafactory.extras import logging as llamafactory_logging

class TrainingMonitor:
    def __init__(self, log_files):
        self.log_files = log_files
        self.setup_loggers()
        
    def setup_loggers(self):
        '''设置日志记录器'''
        self.loggers = {{}}
        
        for log_type, log_file in self.log_files.items():
            logger = logging.getLogger(f"training_{log_type}")
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
            self.loggers[log_type] = logger
            
    def log_training_step(self, step, loss, lr, **kwargs):
        '''记录训练步骤'''
        logger = self.loggers.get('training')
        if logger:
            logger.info(f"🔄 训练步骤 {{step}}")
            logger.info(f"📉 Loss: {{loss:.6f}}")
            logger.info(f"📊 学习率: {{lr:.2e}}")
            
            for key, value in kwargs.items():
                if value is not None:
                    logger.info(f"📈 {{key}}: {{value}}")
                    
    def log_prediction_analysis(self, step, predictions, labels, alignment_score):
        '''记录预测分析'''
        logger = self.loggers.get('predictions')
        if logger:
            logger.info(f"🔮 预测分析 - 步骤 {{step}}")
            logger.info(f"🎯 对齐分数: {{alignment_score:.1f}}%")
            logger.info(f"📊 预测长度: {{len(predictions)}}")
            logger.info(f"📊 标签长度: {{len(labels)}}")
            
    def log_label_analysis(self, step, labels, valid_labels):
        '''记录标签分析'''
        logger = self.loggers.get('labels')
        if logger:
            logger.info(f"🏷️ 标签分析 - 步骤 {{step}}")
            logger.info(f"📊 总标签数: {{len(labels)}}")
            logger.info(f"📊 有效标签数: {{len(valid_labels)}}")
            logger.info(f"📊 有效比例: {{len(valid_labels)/len(labels)*100:.1f}}%")
            
    def log_alignment_analysis(self, step, alignment_data):
        '''记录对齐分析'''
        logger = self.loggers.get('alignment')
        if logger:
            logger.info(f"🎯 对齐分析 - 步骤 {{step}}")
            logger.info(f"📊 对齐数据: {{json.dumps(alignment_data, ensure_ascii=False)}}")

# 创建监控器实例
monitor = TrainingMonitor({log_files})

# 监控训练日志文件
trainer_log = os.path.join("{output_dir}", "trainer_log.jsonl")

def monitor_training():
    '''监控训练过程'''
    print(f"🔍 开始监控训练过程...")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 日志文件: {trainer_log}")
    
    if os.path.exists(trainer_log):
        print(f"✅ 找到训练日志文件: {trainer_log}")
        
        # 监控文件变化
        last_size = 0
        while True:
            try:
                current_size = os.path.getsize(trainer_log)
                if current_size > last_size:
                    # 读取新增内容
                    with open(trainer_log, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_content = f.read()
                        
                    # 解析新日志
                    for line in new_content.strip().split('\\n'):
                        if line.strip():
                            try:
                                log_data = json.loads(line)
                                step = log_data.get('current_steps', 0)
                                loss = log_data.get('loss')
                                lr = log_data.get('lr')
                                
                                if loss is not None:
                                    monitor.log_training_step(step, loss, lr)
                                    
                            except json.JSONDecodeError:
                                continue
                                
                    last_size = current_size
                    
                time.sleep(1)  # 每秒检查一次
                
            except KeyboardInterrupt:
                print("\\n🛑 监控已停止")
                break
            except Exception as e:
                print(f"❌ 监控错误: {{e}}")
                time.sleep(5)
    else:
        print(f"⚠️ 训练日志文件不存在: {trainer_log}")
        print("等待训练开始...")
        
        # 等待文件创建
        while not os.path.exists(trainer_log):
            time.sleep(1)

if __name__ == "__main__":
    monitor_training()
"""
    
    script_path = os.path.join(output_dir, "monitor_training.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(monitoring_script)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    """主函数"""
    print("🚀 启动增强的LLaMA-Factory训练")
    print("=" * 60)
    
    # 创建训练命令
    cmd, output_dir = create_enhanced_training_command()
    print(f"📁 输出目录: {output_dir}")
    
    # 设置日志记录
    log_files = setup_enhanced_logging(output_dir)
    print(f"📝 日志文件:")
    for log_type, log_file in log_files.items():
        print(f"   {log_type}: {log_file}")
    
    # 创建监控脚本
    monitor_script = create_monitoring_script(output_dir, log_files)
    print(f"🔍 监控脚本: {monitor_script}")
    
    # 创建训练脚本
    training_script = os.path.join(output_dir, "run_training.sh")
    with open(training_script, "w", encoding="utf-8") as f:
        f.write(f"""#!/bin/bash
# 增强训练脚本
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 开始增强训练"
echo "📁 输出目录: {output_dir}"
echo "📝 日志文件: {log_files['training']}"
echo "=" * 60

# 启动监控脚本（后台运行）
python3 {monitor_script} &
MONITOR_PID=$!

# 等待一下让监控脚本启动
sleep 2

# 运行训练命令
echo "🔄 执行训练命令..."
{cmd}

# 训练完成后停止监控
echo "🏁 训练完成，停止监控..."
kill $MONITOR_PID 2>/dev/null || true

echo "✅ 训练和监控完成"
""")
    
    # 设置执行权限
    os.chmod(training_script, 0o755)
    
    print(f"📜 训练脚本: {training_script}")
    print("\n🎯 使用方法:")
    print(f"   bash {training_script}")
    print("\n或者直接运行:")
    print(f"   {cmd}")
    
    return training_script, cmd

if __name__ == "__main__":
    training_script, cmd = main()
    
    # 询问是否立即运行
    response = input("\n是否立即开始训练? (y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        print("\n🚀 开始训练...")
        subprocess.run(["bash", training_script])
    else:
        print(f"\n📜 训练脚本已创建: {training_script}")
        print("可以稍后手动运行训练")
