#!/bin/bash
# 最优4卡训练配置
# 基于单卡最佳配置优化，保持训练效果的同时加速训练
# 
# 配置说明：
# - learning_rate: 5e-5（与单卡相同）
# - epochs: 5（与单卡相同）
# - gradient_accumulation: 4（保持有效batch=16，与单卡一致）
# - 其他参数完全匹配最佳单卡配置
# - 预计训练时长：1.5-2小时（vs 单卡6-7小时）

set -e

# 设置GPU（4卡）
export CUDA_VISIBLE_DEVICES=0,2,5,6

# 创建输出目录
TIMESTAMP=$(date '+%Y-%m-%d-%H-%M')
OUTPUT_DIR="/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_${TIMESTAMP}_best_multi"

echo "🚀 启动最优4卡训练（DeepSpeed加速）"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "🎯 目标: 保持最佳训练效果，加速3-4倍"
echo "=" | head -c 60
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 检查DeepSpeed配置文件
if [ ! -f "/home/ziqiang/LLaMA-Factory/ds_config_stage2.json" ]; then
    echo "⚠️  警告: DeepSpeed配置文件不存在，创建默认配置..."
    cat > /home/ziqiang/LLaMA-Factory/ds_config_stage2.json << 'EOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 0.5,
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
EOF
fi

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
    --gradient_accumulation_steps 4 \
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
    --seed 42 \
    --deepspeed /home/ziqiang/LLaMA-Factory/ds_config_stage2.json

echo ""
echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: ${OUTPUT_DIR}"
echo "   📈 训练曲线: ${OUTPUT_DIR}/training_loss.png"
echo "   📝 日志文件: ${OUTPUT_DIR}/trainer_log.jsonl"
echo ""
echo "💡 预期效果："
echo "   - 最终loss: ~0.05-0.10（接近单卡）"
echo "   - 训练时长: ~1.5-2小时（加速3-4倍）"
echo "   - 吞吐量: ~4,300 tokens/s"

