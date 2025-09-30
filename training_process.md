## 训练流程说明：

cd LLaMA-Factory

命令行运行：
'''
CUDA_VISIBLE_DEVICES=4 llamafactory-cli train     --stage sft     --do_train True     --model_name_or_path /data/models/Qwen3-8B     --preprocessing_num_workers 16     --finetuning_type lora     --template qwen3     --flash_attn auto     --dataset_dir data     --dataset mixed_training_data_09_17     --cutoff_len 8192     --learning_rate 5e-05     --num_train_epochs 5     --max_samples 100000     --per_device_train_batch_size 1     --gradient_accumulation_steps 16     --lr_scheduler_type cosine     --max_grad_norm 1.0     --logging_steps 5     --save_steps 100     --warmup_steps 0     --packing False     --enable_thinking False     --overwrite_cache True     --output_dir (替换成自己的路径)    --bf16 True     --plot_loss True     --trust_remote_code True     --ddp_timeout 180000000     --include_num_input_tokens_seen True     --optim adamw_torch     --lora_rank 8     --lora_alpha 16     --lora_dropout 0.1     --lora_target all     --gradient_checkpointing True
'''

对应的data在训练过程中的编码会储存在LLaMA-Factory/sharegpt_pair_debug.log中，方便debug

关于数据集部分：
我这边的数据集LLaMA-Factory/data/dataset/9_17/price_service_train_dataset_v0.json，要训练的时候直接marge就可以

关于我这边的数据的测评，我自己来