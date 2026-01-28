#!/bin/bash
# 最优单卡训练配置
# 基于10-29训练的最佳结果（loss=0.0337）
# 
# 配置说明：
# - learning_rate: 5e-5（经验证的最佳学习率）
# - epochs: 5（避免过拟合，效果最佳）
# - batch_size: 16（有效batch size）
# - 优化器参数：max_grad_norm=0.5, weight_decay=0.01
# - LoRA参数：dropout=0.05, rank=32, alpha=64

set -e

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
TIMESTAMP=$(date '+%Y-%m-%d-%H-%M')
OUTPUT_DIR="/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_${TIMESTAMP}_best_single"

echo "🚀 启动最优单卡训练"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "🎯 目标: 复现最佳训练效果（loss~0.03）"
echo "=" | head -c 60
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行训练
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
    --learning_rate 5.0e-5 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.5 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 200 \
    --warmup_ratio 0.1 \
    --packing False \
    --enable_thinking False \
    --overwrite_cache True \
    --save_strategy steps \
    --output_dir "${OUTPUT_DIR}" \
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
    --dataloader_drop_last True \
    --seed 42

echo ""
echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: ${OUTPUT_DIR}"
echo "   📈 训练曲线: ${OUTPUT_DIR}/training_loss.png"
echo "   📝 日志文件: ${OUTPUT_DIR}/trainer_log.jsonl"
echo ""
echo "💡 预期效果："
echo "   - 最终loss: ~0.03-0.05"
echo "   - 训练时长: ~6-7小时"
echo "   - 吞吐量: ~1,200 tokens/s"

