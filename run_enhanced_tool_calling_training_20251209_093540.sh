#!/bin/bash
# 增强的工具调用训练脚本
# 生成时间: 2025-12-09 09:35:40

set -e

echo "🚀 启动增强的工具调用训练"
echo "📁 输出目录: saves/Qwen3-8B/lora/enhanced_tool_calling_20251209_093540"
echo "=" * 60

# 创建输出目录
mkdir -p "saves/Qwen3-8B/lora/enhanced_tool_calling_20251209_093540"

# 运行训练命令
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset tool_calling_12_08 \
    --cutoff_len 8192 \
    --learning_rate 2e-05 \
    --num_train_epochs 8.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.3 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --packing False \
    --enable_thinking False \
    --overwrite_cache True \
    --output_dir saves/Qwen3-8B/lora/enhanced_tool_calling_20251209_093540 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_target all \
    --gradient_checkpointing True \
    --dataloader_pin_memory False \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --dataloader_drop_last False \
    --seed 42 \
    --save_total_limit 3

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: saves/Qwen3-8B/lora/enhanced_tool_calling_20251209_093540"
