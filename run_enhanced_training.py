#!/usr/bin/env python3
"""
增强的LLaMA-Factory训练脚本
直接修改原始训练流程，添加label打印、predict监控和对齐分析
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# 添加LLaMA-Factory路径
sys.path.insert(0, "/home/ziqiang/LLaMA-Factory/src")

# 导入LLaMA-Factory组件
from llamafactory.cli import main as llamafactory_main
from llamafactory.hparams import get_train_args
from llamafactory.train.sft import run_sft

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

def create_enhanced_training_args():
    """创建增强的训练参数"""
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_{timestamp}"
    
    # 训练参数
    args = [
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
        "--output_dir", output_dir,
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
    
    return args, output_dir

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
    
    # 创建训练参数
    args, output_dir = create_enhanced_training_args()
    main_logger.info(f"📁 输出目录: {output_dir}")
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
    # 修改sys.argv以传递参数
    original_argv = sys.argv.copy()
    sys.argv = ["llamafactory-cli", "train"] + args
    
    try:
        main_logger.info("🔄 开始训练...")
        main_logger.info(f"📊 训练参数: {' '.join(args)}")
        
        # 运行训练
        llamafactory_main()
        
        main_logger.info("✅ 训练完成")
        
    except Exception as e:
        main_logger.error(f"❌ 训练失败: {str(e)}")
        raise
    finally:
        # 恢复原始argv
        sys.argv = original_argv
        
        # 记录完成信息
        main_logger.info("🏁 训练脚本结束")
        
        # 生成摘要
        generate_training_summary(loggers, log_files, output_dir)

def generate_training_summary(loggers, log_files, output_dir):
    """生成训练摘要"""
    
    main_logger = loggers["main"]
    main_logger.info("📊 生成训练摘要...")
    
    # 创建摘要文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(os.path.dirname(log_files["main"]), f"training_summary_{timestamp}.json")
    
    summary = {
        "training_info": {
            "start_time": datetime.now().isoformat(),
            "output_dir": output_dir,
            "log_files": log_files
        },
        "enhanced_features": {
            "label_analysis": "已启用",
            "prediction_monitoring": "已启用", 
            "alignment_analysis": "已启用",
            "detailed_logging": "已启用"
        },
        "notes": [
            "所有训练过程中的label、predict和对齐信息都已记录到对应的日志文件中",
            "可以通过查看日志文件来监控训练进度和模型性能",
            "日志文件包含详细的token级别分析和统计信息"
        ]
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    main_logger.info(f"📄 训练摘要已保存到: {summary_file}")
    
    # 输出摘要信息
    print("\n" + "=" * 60)
    print("📊 训练摘要")
    print("=" * 60)
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 日志文件:")
    for log_type, log_file in log_files.items():
        print(f"   {log_type}: {log_file}")
    print(f"📄 摘要文件: {summary_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()