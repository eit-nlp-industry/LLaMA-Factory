#!/bin/bash

# 使用三段式prompt格式进行评估
# 需要指定训练好的模型路径

MODEL_PATH=${1:-"saves/Qwen3-4B-Instruct-2507/lora/context_enhanced_train_$(date +%Y-%m-%d-%H-%M)"}

echo "开始评估模型: $MODEL_PATH"

CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft \
    --do_eval True \
    --do_predict True \
    --model_name_or_path /data/models/Qwen3-4B-Instruct-2507 \
    --adapter_name_or_path $MODEL_PATH \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset ocr_text_orders_fine_grained_test_08_07_updated_v2_filter_with_context \
    --cutoff_len 2048 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --do_sample False \
    --temperature 0.0 \
    --top_p 1.0 \
    --bf16 True \
    --trust_remote_code True \
    --output_dir ${MODEL_PATH}/eval_results \
    --eval_split test \
    --max_eval_samples 1000 \
    --logging_steps 10
