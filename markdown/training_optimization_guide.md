# 训练优化指南

## 📋 优化总结

根据之前的训练指标分析，我们对训练脚本进行了以下关键优化：

### 🎯 核心优化点

1. **增加保存和评估频率**
   - `SAVE_STEPS`: 500 → **20** (每20步保存一次)
   - `EVAL_STEPS`: 500 → **20** (每20步评估一次)
   - 确保能捕获最佳模型（之前loss最低点在step 40，但只有step 60的checkpoint）

2. **启用最佳模型自动保存**
   - `LOAD_BEST_MODEL_AT_END = True`: 训练结束时自动加载最佳模型
   - `METRIC_FOR_BEST_MODEL = "eval_loss"`: 基于验证集loss选择最佳模型
   - `GREATER_IS_BETTER = False`: loss越小越好

3. **增加正则化防止过拟合**
   - `LORA_DROPOUT`: 0.0 → **0.05** (增加dropout防止过拟合)

4. **增加checkpoint保留数量**
   - `SAVE_TOTAL_LIMIT`: 3 → **10** (保留更多历史checkpoint，便于回退)

## 📊 优化前后对比

| 配置项 | 优化前 | 优化后 | 说明 |
|--------|--------|--------|------|
| SAVE_STEPS | 500 | 20 | 更频繁保存，不错过最佳模型 |
| EVAL_STEPS | 500 | 20 | 更频繁评估，及时监控过拟合 |
| LORA_DROPOUT | 0.0 | 0.05 | 增加正则化 |
| SAVE_TOTAL_LIMIT | 3 | 10 | 保留更多checkpoint |
| LOAD_BEST_MODEL_AT_END | 未设置 | True | 自动保存最佳模型 |
| METRIC_FOR_BEST_MODEL | 未设置 | eval_loss | 基于验证loss选择 |

## 🚀 使用方法

### 方法1: 重新训练（推荐）

如果之前的训练已经完成，建议使用优化后的配置重新训练：

```bash
python tool_calling_train.py
```

### 方法2: 从checkpoint继续训练

如果你想从之前的checkpoint继续训练（例如从checkpoint-60继续）：

```bash
python tool_calling_train.py --resume_from_checkpoint saves/Qwen3-8B/lora/enhanced_tool_calling_XXXXXX/checkpoint-60
```

**注意**: 如果从checkpoint继续训练，需要确保：
- 新的配置会覆盖之前的配置
- 评估数据集必须存在且可用
- 建议重新训练以获得最佳效果

### 方法3: 从最佳loss点继续训练

如果你能找到loss最低点的checkpoint（根据之前的分析，应该在step 40附近），可以从那里继续：

```bash
# 首先需要找到包含step 40的checkpoint目录
# 如果只有checkpoint-60，可能需要手动调整训练步数或重新训练
python tool_calling_train.py --resume_from_checkpoint <checkpoint_path>
```

## 📈 预期效果

使用优化后的配置，你应该能看到：

1. **更频繁的checkpoint保存**
   - 每20步保存一次，包括：checkpoint-20, checkpoint-40, checkpoint-60等
   - 可以轻松回退到任何checkpoint

2. **更及时的过拟合监控**
   - 每20步评估一次，及时发现问题
   - 如果验证loss开始上升，可以提前停止

3. **自动保存最佳模型**
   - 训练结束后，会自动加载验证loss最低的模型
   - 最佳模型会保存在 `output_dir/best_model/` 或直接覆盖最终模型

4. **更好的泛化能力**
   - 通过增加dropout，减少过拟合风险
   - 模型在验证集上的表现应该更稳定

## 🔍 监控训练过程

训练过程中，关注以下指标：

1. **训练loss (loss)**: 应该持续下降
2. **验证loss (eval_loss)**: 应该下降，如果开始上升说明过拟合
3. **学习率 (learning_rate)**: 会随cosine调度器逐渐衰减
4. **梯度范数 (grad_norm)**: 应该在合理范围内（< 1.0）

### 最佳模型判断

- **训练loss最低点**: 可能不是最佳模型（可能过拟合）
- **验证loss最低点**: 这是最佳模型（泛化能力最好）
- 系统会自动选择验证loss最低的模型作为最佳模型

## ⚠️ 注意事项

1. **评估数据集必须存在**
   - 确保 `TEST_DATASET_NAME = "tool_calling_12_08_test"` 在 `dataset_info.json` 中已配置
   - 如果评估数据集不存在，最佳模型保存功能可能无法正常工作

2. **存储空间**
   - 更频繁的保存会占用更多磁盘空间
   - 建议确保有足够的存储空间（至少保留10个checkpoint）

3. **训练时间**
   - 更频繁的评估会增加训练时间
   - 但这是值得的，因为可以及时发现问题并保存最佳模型

4. **从checkpoint继续训练**
   - 如果从checkpoint继续，新的配置参数会生效
   - 建议使用相同的配置继续训练，或者重新训练

## 📝 下一步优化建议

如果训练后仍然发现过拟合，可以考虑：

1. **进一步增加正则化**
   - 将 `LORA_DROPOUT` 增加到 0.1
   - 增加 `WEIGHT_DECAY` 到 0.02

2. **调整学习率调度**
   - 使用线性衰减而不是cosine
   - 设置最小学习率阈值（如果支持）

3. **早停机制**
   - 如果验证loss连续几个epoch不下降，提前停止
   - 可以通过监控训练日志手动停止

4. **数据增强**
   - 增加训练数据的多样性
   - 使用数据增强技术

## 🎓 参考信息

- 之前的训练显示loss在step 40达到最低点（0.1955）
- 之后loss开始上升，说明可能开始过拟合
- 优化后的配置应该能更好地捕获这个最佳点并防止过拟合

