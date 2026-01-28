#!/bin/bash
# 增强的LLaMA-Factory训练脚本
# 生成时间: 2025-12-30 20:10:38

set -e

echo "🚀 启动增强的LLaMA-Factory训练"
echo "📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-12-30-20-10"
echo "📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"
echo "=" * 60

# 设置环境变量 - 双卡DDP训练（保持最佳单卡训练效果）
export CUDA_VISIBLE_DEVICES=0,4

# 创建输出目录
mkdir -p "/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-12-30-20-10"

# 创建日志目录
mkdir -p "/home/ziqiang/LLaMA-Factory/enhanced_training_logs"

# 设置日志文件路径
export ENHANCED_TRAINING_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"
export ENHANCED_LABEL_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20251230_201038.log"
export ENHANCED_PREDICT_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251230_201038.log"
export ENHANCED_ALIGNMENT_LOG="/home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251230_201038.log"

# 记录开始时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 🚀 增强训练开始" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-12-30-20-10" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"

# 检查数据集一致性
echo "🔍 检查数据集一致性..."
python3 /home/ziqiang/LLaMA-Factory/check_dataset_consistency.py >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log" 2>&1
echo "✅ 数据集检查完成"

# 运行训练命令 - 双卡DeepSpeed ZeRO-2分布式训练
echo "🔄 执行双卡DeepSpeed ZeRO-2分布式训练命令..."
echo "⚡ 使用DeepSpeed ZeRO-2，将数据和优化器状态分散到2张卡（共80G显存）"
echo "💾 内存优化策略："
echo "   - DeepSpeed ZeRO-2: 分散优化器状态和梯度到多卡"
echo "   - LoRA rank: 16 (降低从32，减少可训练参数~50%)"
echo "   - gradient_accumulation_steps: 4 (降低从8，减少内存峰值)"
echo "   - gradient_checkpointing: True (启用梯度检查点，用时间换空间)"
echo "   - 优化器: adamw_torch (DeepSpeed兼容)"

# 设置DeepSpeed配置文件路径
DEEPSPEED_CONFIG="/home/ziqiang/LLaMA-Factory/cache/ds_z2_config.json"
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "⚠️ DeepSpeed配置文件不存在，创建默认配置..."
    mkdir -p /home/ziqiang/LLaMA-Factory/cache
    cat > "$DEEPSPEED_CONFIG" << 'EOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 500000000.0,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000.0,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  }
}
EOF
    echo "✅ DeepSpeed配置文件已创建: $DEEPSPEED_CONFIG"
fi

# 设置环境变量以启用DeepSpeed
export FORCE_TORCHRUN=1

llamafactory-cli train     --stage sft     --do_train True     --model_name_or_path /data/models/Qwen3-0.6B     --preprocessing_num_workers 16     --finetuning_type lora     --template qwen     --flash_attn auto     --dataset_dir data     --dataset sft_training_data_filter     --cutoff_len 80000     --learning_rate 5e-5     --num_train_epochs 5.0     --max_samples 100000     --per_device_train_batch_size 1     --gradient_accumulation_steps 4     --lr_scheduler_type cosine     --max_grad_norm 0.5     --weight_decay 0.01     --logging_steps 1     --save_steps 100     --warmup_ratio 0.05     --packing False     --enable_thinking False     --overwrite_cache True     --save_strategy steps     --output_dir /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-12-30-20-10     --bf16 True     --plot_loss True     --trust_remote_code True     --include_num_input_tokens_seen True     --deepspeed "$DEEPSPEED_CONFIG"     --optim adamw_torch     --lora_rank 16     --lora_alpha 32     --lora_dropout 0.05     --lora_target all     --gradient_checkpointing True     --dataloader_pin_memory False     --dataloader_num_workers 0     --remove_unused_columns False     --dataloader_drop_last False     --seed 42

# 记录结束时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | ✅ 训练完成" >> "/home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-12-30-20-10"
echo "   📝 日志文件: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/main_training_20251230_201038.log"
echo "   🏷️ 标签分析: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/label_analysis_20251230_201038.log"
echo "   🔮 预测监控: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/prediction_monitor_20251230_201038.log"
echo "   🎯 对齐分析: /home/ziqiang/LLaMA-Factory/enhanced_training_logs/alignment_analysis_20251230_201038.log"
