#!/usr/bin/env python3
"""
计算 Qwen3-0.6B + LoRA 训练的显存需求
"""
import math

# 模型参数
model_name = "Qwen3-0.6B"
model_params = 0.6e9  # 0.6B 参数
hidden_size = 1024
num_layers = 28
vocab_size = 151936
max_seq_len = 35000  # cutoff_len

# LoRA 配置
lora_rank = 16
lora_alpha = 32
lora_target = "all"  # 所有层

# 训练配置
batch_size = 1
gradient_accumulation = 16
bf16 = True  # bfloat16
gradient_checkpointing = True

# 数据类型大小（字节）
fp32_size = 4
fp16_size = 2
bf16_size = 2

print("=" * 80)
print(f"📊 {model_name} + LoRA 训练显存需求计算")
print("=" * 80)
print(f"\n模型配置:")
print(f"  - 参数量: {model_params/1e9:.2f}B")
print(f"  - Hidden size: {hidden_size}")
print(f"  - 层数: {num_layers}")
print(f"  - Vocab size: {vocab_size}")
print(f"  - 序列长度: {max_seq_len}")

print(f"\nLoRA 配置:")
print(f"  - LoRA rank: {lora_rank}")
print(f"  - LoRA alpha: {lora_alpha}")
print(f"  - LoRA target: {lora_target}")

print(f"\n训练配置:")
print(f"  - Batch size: {batch_size}")
print(f"  - Gradient accumulation: {gradient_accumulation}")
print(f"  - 数据类型: bf16")
print(f"  - Gradient checkpointing: {gradient_checkpointing}")

print("\n" + "=" * 80)
print("显存需求计算:")
print("=" * 80)

# 1. 基础模型参数（bf16）
model_params_memory = model_params * bf16_size / (1024**3)  # GB
print(f"\n1. 基础模型参数 (bf16):")
print(f"   {model_params_memory:.2f} GB")

# 2. LoRA 参数
# LoRA 通常应用于 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# 每层大约 7 个模块，每个模块有 A 和 B 两个矩阵
# A: [hidden_size, lora_rank], B: [lora_rank, hidden_size]
lora_params_per_layer = 7 * 2 * hidden_size * lora_rank  # 7个模块，每个有A和B
total_lora_params = lora_params_per_layer * num_layers
lora_memory = total_lora_params * bf16_size / (1024**3)  # GB
print(f"\n2. LoRA 参数 (bf16):")
print(f"   LoRA 参数量: {total_lora_params/1e6:.2f}M")
print(f"   显存占用: {lora_memory:.2f} GB")

# 3. 优化器状态（AdamW）
# 对于基础模型：需要 momentum 和 variance (fp32)
# 对于 LoRA：也需要 momentum 和 variance (fp32)
optimizer_state_base = model_params * fp32_size * 2 / (1024**3)  # momentum + variance
optimizer_state_lora = total_lora_params * fp32_size * 2 / (1024**3)  # momentum + variance
optimizer_state_total = optimizer_state_base + optimizer_state_lora
print(f"\n3. 优化器状态 (AdamW, fp32):")
print(f"   基础模型: {optimizer_state_base:.2f} GB")
print(f"   LoRA: {optimizer_state_lora:.2f} GB")
print(f"   总计: {optimizer_state_total:.2f} GB")

# 4. 梯度（bf16）
gradient_base = model_params * bf16_size / (1024**3)  # GB
gradient_lora = total_lora_params * bf16_size / (1024**3)  # GB
gradient_total = gradient_base + gradient_lora
print(f"\n4. 梯度 (bf16):")
print(f"   基础模型: {gradient_base:.2f} GB")
print(f"   LoRA: {gradient_lora:.2f} GB")
print(f"   总计: {gradient_total:.2f} GB")

# 5. 激活值（前向传播）
# 这是最关键的！对于长序列，attention 矩阵是 O(seq_len^2)
num_heads = 16  # Qwen3-0.6B 的 attention heads
head_dim = 128

# Attention 激活值（最占内存的部分）
# Q, K, V: batch * num_heads * seq_len * head_dim
qkv_memory = batch_size * num_heads * max_seq_len * head_dim * bf16_size * 3 / (1024**3)  # Q, K, V

# Attention scores: batch * num_heads * seq_len * seq_len (这是最大的！)
attention_scores_memory = batch_size * num_heads * max_seq_len * max_seq_len * bf16_size / (1024**3)

# Attention output: batch * num_heads * seq_len * head_dim
attention_output_memory = batch_size * num_heads * max_seq_len * head_dim * bf16_size / (1024**3)

# MLP 激活值
intermediate_size = hidden_size * 3  # Qwen3 的 intermediate_size
mlp_activation = batch_size * max_seq_len * intermediate_size * bf16_size * 2 / (1024**3)  # gate + up

# 每层的激活值
activation_per_layer = qkv_memory + attention_scores_memory + attention_output_memory + mlp_activation

# 如果使用 gradient checkpointing，只保存部分层的激活值
if gradient_checkpointing:
    # 只保存 checkpoint 点的激活值，大约每 4 层保存一次
    checkpoint_interval = 4
    num_checkpoints = num_layers // checkpoint_interval
    activation_memory = activation_per_layer * num_checkpoints
    print(f"\n5. 激活值 (bf16, gradient_checkpointing=True, checkpoint每{checkpoint_interval}层):")
    print(f"   QKV: {qkv_memory:.2f} GB")
    print(f"   Attention scores (最大): {attention_scores_memory:.2f} GB ⚠️")
    print(f"   Attention output: {attention_output_memory:.2f} GB")
    print(f"   MLP: {mlp_activation:.2f} GB")
    print(f"   每层总计: {activation_per_layer:.2f} GB")
    print(f"   保存 {num_checkpoints} 个 checkpoint: {activation_memory:.2f} GB")
else:
    activation_memory = activation_per_layer * num_layers
    print(f"\n5. 激活值 (bf16, gradient_checkpointing=False):")
    print(f"   QKV: {qkv_memory:.2f} GB")
    print(f"   Attention scores (最大): {attention_scores_memory:.2f} GB ⚠️")
    print(f"   Attention output: {attention_output_memory:.2f} GB")
    print(f"   MLP: {mlp_activation:.2f} GB")
    print(f"   每层: {activation_per_layer:.2f} GB")
    print(f"   {num_layers} 层总计: {activation_memory:.2f} GB")

# 6. 其他开销（临时缓冲区、CUDA等）
other_overhead = 2.0  # GB
print(f"\n6. 其他开销 (CUDA, 临时缓冲区等):")
print(f"   {other_overhead:.2f} GB")

# 总计
total_memory = (
    model_params_memory +
    lora_memory +
    optimizer_state_total +
    gradient_total +
    activation_memory +
    other_overhead
)

print("\n" + "=" * 80)
print("📊 显存需求总计:")
print("=" * 80)
print(f"  基础模型参数:     {model_params_memory:>8.2f} GB")
print(f"  LoRA 参数:         {lora_memory:>8.2f} GB")
print(f"  优化器状态:        {optimizer_state_total:>8.2f} GB")
print(f"  梯度:              {gradient_total:>8.2f} GB")
print(f"  激活值:            {activation_memory:>8.2f} GB")
print(f"  其他开销:          {other_overhead:>8.2f} GB")
print(f"  {'-'*40}")
print(f"  总计:              {total_memory:>8.2f} GB")

print("\n" + "=" * 80)
print("💡 DeepSpeed ZeRO-3 优化:")
print("=" * 80)
print("ZeRO-3 可以将优化器状态、梯度和参数分片到多个GPU:")
print(f"  - 单卡需要: {total_memory:.2f} GB")
print(f"  - 双卡 (ZeRO-3): ~{total_memory/2:.2f} GB/卡")
print(f"  - 四卡 (ZeRO-3): ~{total_memory/4:.2f} GB/卡")

# 计算单卡是否可行
single_gpu_available = 40  # A800 40GB
if total_memory <= single_gpu_available:
    print(f"\n✅ 单卡训练可行 (需要 {total_memory:.2f} GB < {single_gpu_available} GB)")
else:
    print(f"\n❌ 单卡训练不可行 (需要 {total_memory:.2f} GB > {single_gpu_available} GB)")
    print(f"   建议使用 DeepSpeed ZeRO-3 多卡训练")

print("\n" + "=" * 80)

