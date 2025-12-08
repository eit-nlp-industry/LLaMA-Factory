# 快速开始指南

## 一键启动训练

### 最简单的方式

```bash
# 1. 生成所有必要文件
python enhanced_tool_calling_training.py

# 2. 启动训练（自动完成验证和增强）
python run_enhanced_training_complete.py --model_path /data/models/Qwen3-8B
```

### 详细步骤

#### 步骤1: 生成配置文件

```bash
python enhanced_tool_calling_training.py
```

输出：
- ✅ `enhanced_system_prompt.txt` - 增强的系统提示
- ✅ `validate_tool_calling_data.py` - 数据验证工具
- ✅ `run_enhanced_tool_calling_training_*.sh` - 训练脚本
- ✅ 更新 `data/dataset_info.json`

#### 步骤2: 验证数据（可选）

```bash
python validate_tool_calling_data.py data/dataset/12_08/train.json
```

#### 步骤3: 增强数据（推荐）

```bash
python enhance_dataset_with_constraints.py \
    data/dataset/12_08/train.json \
    data/dataset/12_08/train_enhanced.json
```

#### 步骤4: 开始训练

```bash
python run_enhanced_training_complete.py \
    --model_path /data/models/Qwen3-8B \
    --output_dir saves/Qwen3-8B/lora/enhanced_tool_calling
```

## 关键优化点

### 1. 系统提示增强
- 强化 `retrieval_tool` 调用约束
- 明确工具选择规则
- 规范参数提取要求

### 2. 超参数优化
- 学习率: 2.0e-5（稳定学习）
- LoRA rank: 64（提高表达能力）
- 梯度裁剪: 0.3（提高稳定性）
- 训练轮数: 8.0（充分训练）

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

