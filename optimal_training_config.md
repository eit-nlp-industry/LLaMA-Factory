# 🎯 新数据集最佳训练方案

## 📊 数据集信息

- **文件路径**: `/home/ziqiang/LLaMA-Factory/data/dataset/11_07/11.05_train_data_processed.json`
- **样本数量**: 45,610 条
- **数据格式**: ShareGPT格式（conversations）
- **数据集名称**: data_demo

## 🏆 推荐方案：3种配置对比

### 方案A：保守稳定型（强烈推荐⭐⭐⭐）

**适用场景**: 追求最佳效果，数据集质量好

**配置参数**:
```bash
learning_rate: 5.0e-5
num_train_epochs: 5.0
gradient_accumulation_steps: 4 (4卡) / 16 (单卡)
per_device_train_batch_size: 1
effective_batch_size: 16
max_grad_norm: 0.5
weight_decay: 0.01
lora_dropout: 0.05
warmup_ratio: 0.1
lr_scheduler_type: cosine
```

**预期效果**:
- Loss: 0.03-0.05
- 训练稳定，收敛良好
- 泛化能力强

**预计训练时间**:
- 单卡: ~6-7小时
- 4卡: ~1.5-2小时 ✅

---

### 方案B：激进高效型

**适用场景**: 快速迭代，数据集噪声少

**配置参数**:
```bash
learning_rate: 8.0e-5  # ⬆️ 提高60%
num_train_epochs: 4.0   # ⬇️ 减少1个epoch
gradient_accumulation_steps: 4 (4卡) / 16 (单卡)
per_device_train_batch_size: 1
effective_batch_size: 16
max_grad_norm: 0.5
weight_decay: 0.01
lora_dropout: 0.03      # ⬇️ 降低dropout
warmup_ratio: 0.15      # ⬆️ 增加warmup
lr_scheduler_type: cosine
```

**预期效果**:
- Loss: 0.04-0.06
- 训练更快，但需要监控过拟合
- 适合快速实验

**预计训练时间**:
- 单卡: ~5-6小时
- 4卡: ~1.2-1.5小时 ✅

---

### 方案C：深度优化型

**适用场景**: 追求极致效果，数据集质量极高

**配置参数**:
```bash
learning_rate: 3.5e-5   # ⬇️ 降低学习率
num_train_epochs: 8.0   # ⬆️ 增加训练时长
gradient_accumulation_steps: 4 (4卡) / 16 (单卡)
per_device_train_batch_size: 1
effective_batch_size: 16
max_grad_norm: 0.5
weight_decay: 0.02      # ⬆️ 增加正则化
lora_dropout: 0.08      # ⬆️ 增加dropout
warmup_ratio: 0.1
lr_scheduler_type: cosine
```

**预期效果**:
- Loss: 0.02-0.04 (最低)
- 最佳泛化能力
- 训练时间最长

**预计训练时间**:
- 单卡: ~10-11小时
- 4卡: ~2.5-3小时

---

## 🎯 最终推荐：方案A（保守稳定型）

**理由**:
1. ✅ 基于历史最佳配置（loss=0.0337）
2. ✅ 稳定性高，不易过拟合
3. ✅ 4卡训练时间合理（~2小时）
4. ✅ 适合新数据集的首次训练

## 🔧 配置文件生成

### 单卡训练配置（如果时间充裕）

```bash
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
    --output_dir "saves/Qwen3-8B/lora/train_optimal_single" \
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
```

### 4卡DeepSpeed训练配置（推荐✅）

```bash
# 环境变量
export CUDA_VISIBLE_DEVICES=0,2,5,6

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
    --output_dir "saves/Qwen3-8B/lora/train_optimal_4gpu" \
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
```

## 📈 训练监控要点

### 关键指标监控

1. **Loss曲线**
   - 应该平滑下降
   - 5 epochs后应低于0.1
   - 最终loss目标: 0.03-0.05

2. **学习率曲线**
   - Warmup阶段: 0 → 5.0e-5 (前10%步数)
   - Cosine衰减: 5.0e-5 → 0
   - 确保曲线平滑

3. **Gradient Norm**
   - 应该在0.5以下
   - 如果经常达到上限，考虑增加max_grad_norm

4. **训练速度**
   - 单卡: ~1,200-1,300 tokens/s
   - 4卡: ~4,300-4,500 tokens/s

### 何时停止训练

1. **正常情况**: 完成5 epochs
2. **提前停止**: 如果loss在3-4 epochs已经收敛
3. **延长训练**: 如果5 epochs后loss还在下降

## 🔬 进阶优化建议

### 如果训练效果不理想

1. **Loss下降缓慢**
   - 尝试增加学习率到 8.0e-5
   - 检查数据质量
   - 减少weight_decay到0.005

2. **Loss震荡**
   - 降低学习率到 3.5e-5
   - 增加warmup_ratio到0.15
   - 检查batch size是否过小

3. **过拟合迹象**
   - 增加lora_dropout到0.08
   - 增加weight_decay到0.02
   - 减少训练epochs

### LoRA参数调优

当前配置已经很好，但如果需要调整：

```python
# 标准配置（当前）
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05

# 更强能力（如果数据复杂）
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

# 更快训练（如果数据简单）
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.03
```

## 🎯 实验流程建议

### 第一步：基线训练（必做）

使用方案A配置，4卡训练：
```bash
python3 /home/ziqiang/LLaMA-Factory/create_enhanced_training.py
# 然后运行生成的训练脚本
```

**成功标准**:
- ✅ Loss < 0.1 (5 epochs)
- ✅ 训练稳定，无异常
- ✅ 模型可以正常推理

### 第二步：效果评估（必做）

1. **定量评估**
   - 记录最终loss值
   - 检查训练曲线
   - 计算训练时长

2. **定性评估**
   - 测试模型推理效果
   - 对比训练前后差异
   - 检查是否有过拟合

### 第三步：参数调优（可选）

基于第一步的结果，考虑：

| 问题 | 解决方案 |
|-----|---------|
| Loss偏高 | 尝试方案C（更多epochs） |
| 训练太慢 | 尝试方案B（更高学习率） |
| 效果很好 | 保持当前配置 ✅ |

## 📝 配置对比表

| 参数 | 方案A（推荐）| 方案B（快速）| 方案C（深度）|
|-----|------------|------------|------------|
| learning_rate | 5.0e-5 | 8.0e-5 | 3.5e-5 |
| num_epochs | 5.0 | 4.0 | 8.0 |
| lora_dropout | 0.05 | 0.03 | 0.08 |
| weight_decay | 0.01 | 0.01 | 0.02 |
| warmup_ratio | 0.1 | 0.15 | 0.1 |
| 预期loss | 0.03-0.05 | 0.04-0.06 | 0.02-0.04 |
| 训练时间(4卡) | 2h | 1.5h | 3h |

## ⚡ 快速开始

### 方式1：使用现有脚本（推荐）

```bash
cd /home/ziqiang/LLaMA-Factory
python3 create_enhanced_training.py
# 按提示运行生成的训练脚本
```

### 方式2：直接运行命令

```bash
cd /home/ziqiang/LLaMA-Factory

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,2,5,6

# 运行训练（4卡）
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
    --output_dir "saves/Qwen3-8B/lora/train_optimal" \
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
    --deepspeed ds_config_stage2.json
```

## 🎉 预期结果

使用方案A配置，在新数据集上训练后：

- ✅ **Loss**: 0.03-0.05（5 epochs）
- ✅ **训练时间**: ~2小时（4卡）
- ✅ **模型效果**: 接近或超过历史最佳
- ✅ **稳定性**: 训练过程平滑，无异常

---

**生成时间**: 2025-11-08  
**数据集**: 11.05_train_data_processed.json (45,610样本)  
**推荐配置**: 方案A（保守稳定型）⭐

