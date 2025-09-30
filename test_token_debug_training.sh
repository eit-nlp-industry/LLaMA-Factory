#!/bin/bash

# 测试token调试功能的小规模训练脚本
echo "🚀 开始测试token调试功能..."

# 清理之前的日志
rm -f /home/ziqiang/LLaMA-Factory/token_debug_current.log

echo "📝 运行小规模测试训练..."

CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 1 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mixed_training_data_09_17 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --max_samples 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --output_dir /tmp/test_token_debug \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --lora_target all \
    --overwrite_output_dir \
    2>&1 | tee test_token_debug.log

echo ""
echo "✅ 训练完成，检查日志..."

# 检查调试日志
if [ -f "/home/ziqiang/LLaMA-Factory/token_debug_current.log" ]; then
    echo "🎉 找到token调试日志文件："
    echo "📄 /home/ziqiang/LLaMA-Factory/token_debug_current.log"
    echo ""
    echo "📊 日志内容预览："
    head -20 /home/ziqiang/LLaMA-Factory/token_debug_current.log
    echo ""
    echo "📈 日志统计："
    echo "TOKEN_DEBUG 行数: $(grep -c "TOKEN_DEBUG" /home/ziqiang/LLaMA-Factory/token_debug_current.log)"
    echo "TEMPLATE_DEBUG 行数: $(grep -c "TEMPLATE_DEBUG" /home/ziqiang/LLaMA-Factory/token_debug_current.log)"
    echo "INFER_SEQLEN 行数: $(grep -c "INFER_SEQLEN" /home/ziqiang/LLaMA-Factory/token_debug_current.log)"
else
    echo "❌ 未找到token调试日志文件"
fi

# 检查训练日志中的DEBUG信息
echo ""
echo "🔍 训练日志中的DEBUG信息："
grep -c "DEBUG |" test_token_debug.log || echo "未找到DEBUG信息"

echo ""
echo "🏁 测试完成！"
