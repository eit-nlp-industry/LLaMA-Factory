#!/bin/bash

# 激活conda环境
source /home/ziqiang/.conda/envs/llama_factory/bin/activate

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=5,6

# 运行修正后的训练命令
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-0.6B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ocr_text_orders \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 10.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir saves/Qwen3-0.6B/lora/train_2025-08-01-10-44-05 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all 