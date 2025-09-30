#!/bin/bash
# 增强的LLaMA-Factory训练脚本
# 生成时间: 2025-09-27 22:42:04

set -e

echo "🚀 启动增强的LLaMA-Factory训练"
echo "📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-27-22-42"
echo "📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"
echo "=" * 60

# 设置环境变量
export CUDA_VISIBLE_DEVICES=6

# 创建输出目录
mkdir -p "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-27-22-42"

# 创建日志目录
mkdir -p "/home/ziqiang/LLaMA-Factory/enhanced_training_logs"

# 设置日志文件路径
export ENHANCED_TRAINING_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"
export ENHANCED_LABEL_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20250927_224204.log"
export ENHANCED_PREDICT_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20250927_224204.log"
export ENHANCED_ALIGNMENT_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20250927_224204.log"

# 记录开始时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 🚀 增强训练开始" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-27-22-42" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"

# 运行训练命令
echo "🔄 执行训练命令..."
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset data_demo \
    --cutoff_len 8192 \
    --learning_rate 5.0e-4 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 50 \
    --warmup_steps 100 \
    --packing False \
    --enable_thinking False \
    --overwrite_cache True \
    --save_strategy epoch \
    --save_total_limit 2 \
    --output_dir "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-27-22-42" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target all \
    --gradient_checkpointing True

# 记录结束时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | ✅ 训练完成" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-27-22-42"
echo "   📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20250927_224204.log"
echo "   🏷️ 标签分析: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20250927_224204.log"
echo "   🔮 预测监控: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20250927_224204.log"
echo "   🎯 对齐分析: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20250927_224204.log"
