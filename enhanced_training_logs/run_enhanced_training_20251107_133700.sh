#!/bin/bash
# 增强的LLaMA-Factory训练脚本
# 生成时间: 2025-11-07 13:37:00

set -e

echo "🚀 启动增强的LLaMA-Factory训练"
echo "📁 输出目录: /home/qiyang_shi/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-07-13-37"
echo "📝 日志文件: /home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"
echo "=" * 60

# 设置环境变量 - 先使用单GPU 6
export CUDA_VISIBLE_DEVICES=6

# 创建输出目录
mkdir -p "/home/qiyang_shi/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-07-13-37"

# 创建日志目录
mkdir -p "/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs"

# 设置日志文件路径
export ENHANCED_TRAINING_LOG="/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"
export ENHANCED_LABEL_LOG="/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/label_analysis_20251107_133700.log"
export ENHANCED_PREDICT_LOG="/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251107_133700.log"
export ENHANCED_ALIGNMENT_LOG="/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251107_133700.log"

# 记录开始时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 🚀 增强训练开始" >> "/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📁 输出目录: /home/qiyang_shi/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-07-13-37" >> "/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📝 日志文件: /home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log" >> "/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"

# 运行训练命令 - 多GPU分布式训练
echo "🔄 执行多GPU分布式训练命令..."
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 1 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset data_demo \
    --cutoff_len 10240 \
    --learning_rate 5.0e-5 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --logging_steps 5 \
    --save_steps 50 \
    --warmup_ratio 0.1 \
    --packing False \
    --enable_thinking False \
    --overwrite_cache True \
    --save_strategy steps \
    --output_dir "/home/qiyang_shi/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-07-13-37" \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target all \
    --gradient_checkpointing True \
    --dataloader_pin_memory False \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --dataloader_drop_last True

# 记录结束时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | ✅ 训练完成" >> "/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: /home/qiyang_shi/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-07-13-37"
echo "   📝 日志文件: /home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/main_training_20251107_133700.log"
echo "   🏷️ 标签分析: /home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/label_analysis_20251107_133700.log"
echo "   🔮 预测监控: /home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251107_133700.log"
echo "   🎯 对齐分析: /home/qiyang_shi/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251107_133700.log"
