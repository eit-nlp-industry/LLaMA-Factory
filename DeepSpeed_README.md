# DeepSpeed 配置说明

## 📋 概述

本项目提供了两个 DeepSpeed 配置文件，用于加速 LLaMA-Factory 训练：

- `ds_config_stage2.json` - **推荐使用** ✨
- `ds_config_stage3.json` - 极致内存优化

## 🚀 DeepSpeed Stage 2 (推荐)

### 特点
- ✅ **优化器状态分片**：将优化器状态分布到多个 GPU
- ✅ **CPU Offload**：将优化器状态卸载到 CPU 内存，节省 GPU 显存
- ✅ **通信优化**：overlap_comm 和 allgather 优化
- ✅ **稳定性好**：适合大多数训练场景
- ✅ **速度快**：相比普通 DDP 有明显加速

### 适用场景
- 模型参数量：8B - 13B
- GPU 数量：2-8 卡
- 显存要求：每卡 24GB+

### 配置文件
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

## 💾 DeepSpeed Stage 3 (极致优化)

### 特点
- ✅ **模型参数分片**：将模型参数也分布到多个 GPU
- ✅ **CPU Offload**：优化器和参数都可以卸载到 CPU
- ✅ **极致内存节省**：可以训练更大的模型
- ⚠️ **速度较慢**：通信开销更大
- ⚠️ **需要更多调试**：配置复杂度更高

### 适用场景
- 模型参数量：13B+
- GPU 数量：4-16 卡
- 显存要求：每卡 16GB+
- 显存紧张时的最佳选择

### 配置文件
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu"
    },
    "offload_param": {
      "device": "cpu"
    }
  }
}
```

## 🔧 使用方法

### 1. 使用 Stage 2（默认）
```bash
--deepspeed /home/ziqiang/LLaMA-Factory/ds_config_stage2.json
```

### 2. 切换到 Stage 3
在 `create_enhanced_training.py` 中修改：
```python
--deepspeed /home/ziqiang/LLaMA-Factory/ds_config_stage3.json
```

## 📊 性能对比

| 配置 | 训练速度 | 显存占用 | 稳定性 | 推荐场景 |
|------|---------|---------|--------|---------|
| **普通 DDP** | 基准 | 高 | 高 | 小模型 |
| **Stage 2** | 1.5-2x | 中 | 高 | 大多数场景 ✨ |
| **Stage 3** | 1.2-1.5x | 低 | 中 | 显存受限 |

## ⚙️ 训练参数说明

当前训练配置（配合 10 epochs）：

```bash
--num_train_epochs 10.0              # 训练轮数
--per_device_train_batch_size 1      # 每卡批次大小
--gradient_accumulation_steps 16     # 梯度累积步数
--learning_rate 3.5e-5               # 学习率
--lr_scheduler_type cosine           # 余弦学习率调度
--warmup_ratio 0.1                   # 预热比例 (10%)
--max_grad_norm 0.4                  # 梯度裁剪
--weight_decay 0.02                  # 权重衰减
```

### 有效批次大小计算
```
总批次大小 = per_device_batch_size × GPU数量 × gradient_accumulation_steps
         = 1 × 4 × 16 = 64
```

## 🎯 GPU 配置

### 当前配置（4 卡）
```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

### 调整 GPU 数量
- **2 卡**：`CUDA_VISIBLE_DEVICES=4,5`
- **8 卡**：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

⚠️ **注意**：调整 GPU 数量后，可能需要相应调整 `gradient_accumulation_steps` 以保持总批次大小不变。

## 🐛 常见问题

### 1. OOM (Out of Memory)
**解决方案**：
- 切换到 Stage 3
- 减少 `per_device_train_batch_size` 到 0.5
- 增加 `gradient_accumulation_steps`

### 2. 训练速度慢
**解决方案**：
- 使用 Stage 2 代替 Stage 3
- 增加 GPU 数量
- 减少 `gradient_accumulation_steps`（如果显存允许）

### 3. 进程卡住不动
**解决方案**：
- 检查网络连接（多机训练时）
- 增加 `ddp_timeout`
- 检查 DeepSpeed 版本兼容性

## 📦 环境要求

```bash
pip install deepspeed>=0.10.0
pip install transformers>=4.30.0
pip install torch>=2.0.0
```

## 💡 优化建议

### 针对 10 Epochs 训练
1. **学习率调度**：cosine 非常适合长时间训练
2. **保存策略**：
   - `save_steps 200`：适中的保存频率
   - 10 epochs 可以考虑增加到 500 以节省存储
3. **监控指标**：
   - 每 1 步记录日志（`logging_steps 1`）
   - 关注学习率曲线和 loss 变化

### 显存优化技巧
1. ✅ 已启用 `gradient_checkpointing`
2. ✅ 已使用 `bf16` 混合精度
3. ✅ 已禁用 `dataloader_pin_memory`（与 CPU offload 配合更好）

## 📞 参考资源

- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [ZeRO 优化器论文](https://arxiv.org/abs/1910.02054)
- [LLaMA-Factory 文档](https://github.com/hiyouga/LLaMA-Factory)

---

**推荐配置**：对于 Qwen3-8B + 10 epochs 训练，使用 **Stage 2** 配置即可获得最佳性能平衡！✨



