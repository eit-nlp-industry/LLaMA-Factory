# 工具调用训练快速开始

## 一键启动

```bash
# 1. 首次使用：生成辅助工具（只需一次）
python tool_calling_setup.py

# 2. 直接训练（自动完成验证和增强）
python tool_calling_train.py
```

## 配置说明

在 `tool_calling_train.py` 顶部修改超参数：
- `CUDA_VISIBLE_DEVICES`: GPU设备（如 "0" 或 "0,2"）
- `MODEL_PATH`: 模型路径
- `LEARNING_RATE`: 学习率（默认3.5e-5）
- `LORA_RANK`: LoRA rank（默认32）
- 其他参数见脚本注释

## 其他工具

```bash
# 单独增强数据
python tool_calling_enhance_data.py data/dataset/12_08/train.json output.json

# 单独验证数据
python validate_tool_calling_data.py data/dataset/12_08/train.json
```

### 3. 约束机制
- ✅ 强制检索阶段调用 `retrieval_tool`
- ✅ 限制工具选择范围（top5）
- ✅ 严格参数提取验证
- ✅ 结果总结质量要求

## 预期训练时间

- **单GPU (A100)**: 约 6-8 小时
- **双GPU (DDP)**: 约 3-4 小时
- **四GPU (DDP)**: 约 1.5-2 小时

## 训练后检查

训练完成后，检查：
1. 训练loss是否稳定下降
2. 验证指标是否提升
3. 模型是否遵循约束规则

## 常见问题

**Q: 如何修改模型路径？**  
A: 使用 `--model_path` 参数：
```bash
python run_enhanced_training_complete.py --model_path /your/model/path
```

**Q: 如何跳过数据增强？**  
A: 使用 `--skip_enhancement` 参数：
```bash
python run_enhanced_training_complete.py --skip_enhancement
```

**Q: 如何只查看命令不执行？**  
A: 使用 `--dry_run` 参数：
```bash
python run_enhanced_training_complete.py --dry_run
```

## 下一步

训练完成后：
1. 在测试集上评估模型性能
2. 检查模型是否遵循约束规则
3. 根据结果调整超参数
4. 迭代优化

