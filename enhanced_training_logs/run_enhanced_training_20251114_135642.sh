#!/bin/bash
# 增强的LLaMA-Factory训练脚本
# 生成时间: 2025-11-14 13:56:42

set -e

echo "🚀 启动增强的LLaMA-Factory训练"
echo "📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-14-13-56"
echo "📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"
echo "=" * 60

# 设置环境变量 - 双卡DDP训练（保持最佳单卡训练效果）
export CUDA_VISIBLE_DEVICES=0,2

# 创建输出目录
mkdir -p "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-14-13-56"

# 创建日志目录
mkdir -p "/home/ziqiang/LLaMA-Factory/enhanced_training_logs"

# 设置日志文件路径
export ENHANCED_TRAINING_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"
export ENHANCED_LABEL_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20251114_135642.log"
export ENHANCED_PREDICT_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251114_135642.log"
export ENHANCED_ALIGNMENT_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251114_135642.log"

# 记录开始时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 🚀 增强训练开始" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-14-13-56" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"

# 检查数据集一致性
echo "🔍 检查数据集一致性..."
python3 /home/ziqiang/LLaMA-Factory/check_dataset_consistency.py >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log" 2>&1
echo "✅ 数据集检查完成"

# 运行训练命令 - 双卡DDP分布式训练
echo "🔄 执行双卡DDP分布式训练命令..."
echo "⚡ 使用PyTorch DDP，配置与最佳单卡完全等效"
llamafactory-cli train     --stage sft     --do_train True     --model_name_or_path /data/models/Qwen3-8B     --preprocessing_num_workers 16     --finetuning_type lora     --template qwen3     --flash_attn auto     --dataset_dir data     --dataset data_demo     --cutoff_len 8192     --learning_rate 3.5e-5     --num_train_epochs 10.0     --max_samples 100000     --per_device_train_batch_size 1     --gradient_accumulation_steps 8     --lr_scheduler_type cosine     --max_grad_norm 0.5     --weight_decay 0.01     --logging_steps 1     --save_steps 100     --warmup_ratio 0.05     --packing False     --enable_thinking False     --overwrite_cache True     --save_strategy steps     --output_dir /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-14-13-56     --bf16 True     --plot_loss True     --trust_remote_code True     --ddp_timeout 180000000     --include_num_input_tokens_seen True     --optim adamw_torch     --lora_rank 32     --lora_alpha 64     --lora_dropout 0.05     --lora_target all     --gradient_checkpointing True     --dataloader_pin_memory False     --dataloader_num_workers 0     --remove_unused_columns False     --dataloader_drop_last False     --seed 42

# 记录结束时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | ✅ 训练完成" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-11-14-13-56"
echo "   📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251114_135642.log"
echo "   🏷️ 标签分析: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20251114_135642.log"
echo "   🔮 预测监控: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251114_135642.log"
echo "   🎯 对齐分析: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251114_135642.log"
