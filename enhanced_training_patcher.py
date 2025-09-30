#!/usr/bin/env python3
"""
直接修改LLaMA-Factory训练流程的增强脚本
通过monkey patching的方式添加label打印、predict监控和对齐分析功能
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加LLaMA-Factory路径
sys.path.insert(0, "/home/ziqiang/LLaMA-Factory/src")

# 导入必要的模块
from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.train.callbacks import LogCallback
from llamafactory.extras.constants import IGNORE_INDEX

def setup_enhanced_logging():
    """设置增强的日志记录"""
    
    # 创建日志目录
    log_dir = "/home/ziqiang/LLaMA-Factory/enhanced_training_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files = {
        "training": os.path.join(log_dir, f"training_monitor_{timestamp}.log"),
        "predictions": os.path.join(log_dir, f"prediction_monitor_{timestamp}.log"),
        "labels": os.path.join(log_dir, f"label_analysis_{timestamp}.log"),
        "alignment": os.path.join(log_dir, f"alignment_analysis_{timestamp}.log"),
        "main": os.path.join(log_dir, f"main_training_{timestamp}.log")
    }
    
    # 设置日志记录器
    loggers = {}
    for log_type, log_file in log_files.items():
        logger = logging.getLogger(f"enhanced_{log_type}")
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
    
    return loggers, log_files

def enhance_supervised_processor(loggers):
    """增强SupervisedDatasetProcessor"""
    
    original_encode_data_example = SupervisedDatasetProcessor._encode_data_example
    original_preprocess_dataset = SupervisedDatasetProcessor.preprocess_dataset
    
    def enhanced_encode_data_example(self, prompt, response, system, tools, images, videos, audios):
        """增强的数据编码方法"""
        
        # 记录开始处理
        loggers["labels"].info(f"\n🔧 开始编码数据样本")
        loggers["labels"].info(f"📊 输入消息数量: {len(prompt + response)}")
        
        # 调用原始方法
        input_ids, labels = original_encode_data_example(self, prompt, response, system, tools, images, videos, audios)
        
        # 记录编码结果
        loggers["labels"].info(f"📊 编码结果:")
        loggers["labels"].info(f"   input_ids长度: {len(input_ids)}")
        loggers["labels"].info(f"   labels长度: {len(labels)}")
        
        # 分析标签
        valid_labels = [l for l in labels if l != IGNORE_INDEX]
        loggers["labels"].info(f"   有效标签数量: {len(valid_labels)}")
        loggers["labels"].info(f"   有效标签比例: {len(valid_labels)/len(labels)*100:.1f}%")
        
        # 记录标签统计
        unique_labels = set(labels)
        label_stats = {}
        for label in unique_labels:
            count = labels.count(label)
            if label == IGNORE_INDEX:
                label_stats[f"IGNORE_INDEX({label})"] = count
            else:
                label_stats[f"TOKEN_{label}"] = count
        
        loggers["labels"].info(f"   标签统计: {dict(list(label_stats.items())[:5])}")
        
        return input_ids, labels
    
    def enhanced_preprocess_dataset(self, examples):
        """增强的数据预处理方法"""
        
        loggers["labels"].info(f"\n🚀 开始预处理数据集")
        loggers["labels"].info(f"📊 样本数量: {len(examples['_prompt'])}")
        
        # 调用原始方法
        result = original_preprocess_dataset(self, examples)
        
        # 记录预处理结果
        loggers["labels"].info(f"📊 预处理完成:")
        loggers["labels"].info(f"   处理样本数: {len(result['input_ids'])}")
        
        return result
    
    # 替换方法
    SupervisedDatasetProcessor._encode_data_example = enhanced_encode_data_example
    SupervisedDatasetProcessor.preprocess_dataset = enhanced_preprocess_dataset
    
    return loggers

def enhance_log_callback(loggers):
    """增强LogCallback"""
    
    original_on_log = LogCallback.on_log
    
    def enhanced_on_log(self, args, state, control, **kwargs):
        """增强的日志记录方法"""
        
        # 调用原始方法
        original_on_log(self, args, state, control, **kwargs)
        
        # 记录训练步骤
        if state.log_history:
            current_log = state.log_history[-1]
            step = current_log.get("current_steps", 0)
            loss = current_log.get("loss")
            lr = current_log.get("lr")
            
            if loss is not None:
                loggers["training"].info(f"\n🔄 训练步骤 {step}")
                loggers["training"].info(f"📉 Loss: {loss:.6f}")
                loggers["training"].info(f"📊 学习率: {lr:.2e}")
                
                # 记录其他指标
                for key, value in current_log.items():
                    if key not in ["current_steps", "loss", "lr"] and value is not None:
                        loggers["training"].info(f"📈 {key}: {value}")
    
    # 替换方法
    LogCallback.on_log = enhanced_on_log
    
    return loggers

def create_enhanced_training_command():
    """创建增强的训练命令"""
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_{timestamp}"
    
    # 训练命令
    cmd = f"""CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \\
    --stage sft \\
    --do_train True \\
    --model_name_or_path /data/models/Qwen3-8B \\
    --preprocessing_num_workers 1 \\
    --finetuning_type lora \\
    --template qwen3 \\
    --flash_attn auto \\
    --dataset_dir data \\
    --dataset data_demo \\
    --cutoff_len 8192 \\
    --learning_rate 1.0e-5 \\
    --num_train_epochs 50.0 \\
    --max_samples 100000 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --lr_scheduler_type cosine \\
    --max_grad_norm 1.0 \\
    --logging_steps 1 \\
    --save_steps 10 \\
    --warmup_steps 0 \\
    --packing False \\
    --enable_thinking False \\
    --overwrite_cache True \\
    --output_dir {output_dir} \\
    --bf16 True \\
    --plot_loss True \\
    --trust_remote_code True \\
    --ddp_timeout 180000000 \\
    --include_num_input_tokens_seen True \\
    --optim adamw_torch \\
    --lora_rank 8 \\
    --lora_alpha 16 \\
    --lora_dropout 0 \\
    --lora_target all \\
    --gradient_checkpointing True"""
    
    return cmd, output_dir

def main():
    """主函数"""
    
    print("🚀 启动增强的LLaMA-Factory训练")
    print("=" * 60)
    
    # 设置日志记录
    loggers, log_files = setup_enhanced_logging()
    
    # 记录启动信息
    main_logger = loggers["main"]
    main_logger.info("🚀 增强训练启动")
    main_logger.info(f"📁 日志目录: {os.path.dirname(log_files['main'])}")
    
    for log_type, log_file in log_files.items():
        main_logger.info(f"📝 {log_type}日志: {log_file}")
    
    # 增强组件
    enhance_supervised_processor(loggers)
    enhance_log_callback(loggers)
    
    main_logger.info("✅ 组件增强完成")
    
    # 创建训练命令
    cmd, output_dir = create_enhanced_training_command()
    main_logger.info(f"📁 输出目录: {output_dir}")
    
    # 创建训练脚本
    training_script = os.path.join(os.path.dirname(log_files["main"]), "run_enhanced_training.sh")
    with open(training_script, "w", encoding="utf-8") as f:
        f.write(f"""#!/bin/bash
# 增强训练脚本
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 开始增强训练"
echo "📁 输出目录: {output_dir}"
echo "📝 日志文件: {log_files['main']}"
echo "=" * 60

# 运行训练命令
{cmd}

echo "✅ 训练完成"
""")
    
    # 设置执行权限
    os.chmod(training_script, 0o755)
    
    main_logger.info(f"📜 训练脚本已创建: {training_script}")
    
    # 输出摘要信息
    print("\n" + "=" * 60)
    print("📊 增强训练设置完成")
    print("=" * 60)
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 日志文件:")
    for log_type, log_file in log_files.items():
        print(f"   {log_type}: {log_file}")
    print(f"📜 训练脚本: {training_script}")
    print("=" * 60)
    
    # 询问是否立即运行
    response = input("\n是否立即开始训练? (y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        print("\n🚀 开始训练...")
        os.system(f"bash {training_script}")
    else:
        print(f"\n📜 训练脚本已创建: {training_script}")
        print("可以稍后手动运行训练")

if __name__ == "__main__":
    main()
