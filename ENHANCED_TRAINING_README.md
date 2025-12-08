# 增强的工具调用训练方案

## 概述

本方案旨在从模板层面强化模型的任务执行能力，通过约束机制确保模型：
1. **检索阶段**：必须正确调用 `retrieval_tool`
2. **工具选择**：必须从 `retrieval_tool` 返回的 top5 工具中选择
3. **参数提取**：严格按照 `inputSchema` 和 `query` 提取合适的参数
4. **结果总结**：给出准确完整的总结
5. **避免预期外预测**：通过约束机制减少错误预测

## 文件说明

### 核心文件

1. **`enhanced_tool_calling_training.py`**
   - 主配置脚本
   - 生成增强的系统提示、训练配置、训练脚本和数据验证工具

2. **`run_enhanced_training_complete.py`**
   - 完整的训练启动脚本
   - 整合所有功能，一键启动训练
   - 支持数据验证、数据增强、训练执行

3. **`enhance_dataset_with_constraints.py`**
   - 数据增强脚本
   - 将增强的系统提示应用到训练数据
   - 验证数据是否符合约束要求

4. **`validate_tool_calling_data.py`** (自动生成)
   - 数据验证工具
   - 检查数据是否符合工具调用约束

5. **`enhanced_system_prompt.txt`** (自动生成)
   - 增强的系统提示模板
   - 包含详细的约束规则和流程说明

## 快速开始

### 步骤1: 生成所有必要文件

```bash
python enhanced_tool_calling_training.py
```

这将生成：
- `enhanced_system_prompt.txt` - 增强的系统提示
- `validate_tool_calling_data.py` - 数据验证工具
- `run_enhanced_tool_calling_training_*.sh` - 训练脚本

### 步骤2: 验证训练数据

```bash
python validate_tool_calling_data.py data/dataset/12_08/train.json
```

### 步骤3: 增强训练数据（可选但推荐）

```bash
python enhance_dataset_with_constraints.py \
    data/dataset/12_08/train.json \
    data/dataset/12_08/train_enhanced.json
```

### 步骤4: 启动训练

#### 方式1: 使用完整启动脚本（推荐）

```bash
python run_enhanced_training_complete.py \
    --model_path /data/models/Qwen3-8B \
    --output_dir saves/Qwen3-8B/lora/enhanced_tool_calling
```

#### 方式2: 使用生成的Shell脚本

```bash
bash run_enhanced_tool_calling_training_*.sh
```

## 优化配置说明

### 超参数优化

本方案针对工具调用任务优化了以下超参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 2.0e-5 | 适中的学习率，确保稳定学习 |
| `num_train_epochs` | 8.0 | 足够的轮数但不过度训练 |
| `per_device_train_batch_size` | 1 | 单设备批次大小 |
| `gradient_accumulation_steps` | 16 | 有效batch size = 16 |
| `warmup_ratio` | 0.1 | 10%的warmup，帮助稳定训练 |
| `max_grad_norm` | 0.3 | 更严格的梯度裁剪，提高稳定性 |
| `lora_rank` | 64 | 更高的rank，提高表达能力 |
| `lora_alpha` | 128 | alpha = 2 * rank，保持比例 |
| `lora_dropout` | 0.1 | 适度的dropout，防止过拟合 |
| `weight_decay` | 0.01 | 权重衰减，防止过拟合 |

### 约束机制

#### 1. 检索阶段约束

- **强制要求**：必须首先调用 `retrieval_tool`
- **参数要求**：
  - `query`: 用户的完整查询内容
  - `source_filter`: 必须为 `"toollist"`
  - `top_k`: 建议设置为 5
  - `user_id`: 有效的用户ID

#### 2. 工具选择约束

- **来源限制**：只能从 `retrieval_tool` 返回的 top5 工具列表中选择
- **选择标准**：
  - 工具描述与查询意图匹配
  - 工具类别（listing/analysis）适合查询类型
  - 工具的 `inputSchema` 支持所需参数

#### 3. 参数提取约束

- **严格遵循**：必须按照 `inputSchema` 定义提取参数
- **必填参数**：所有 `required` 字段必须提供
- **类型匹配**：参数类型必须符合定义
- **格式要求**：时间参数使用 YYYY-MM-DD 格式

#### 4. 结果总结约束

- **准确性**：基于工具返回结果生成
- **完整性**：包含关键信息
- **错误处理**：明确说明空数据或错误原因

## 数据格式要求

训练数据应为 ShareGPT 格式：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "用户查询"
    },
    {
      "from": "function_call",
      "value": "{\"name\": \"retrieval_tool\", \"arguments\": {...}}"
    },
    {
      "from": "observation",
      "value": "[工具列表JSON]"
    },
    {
      "from": "function_call",
      "value": "{\"name\": \"业务工具\", \"arguments\": {...}}"
    },
    {
      "from": "observation",
      "value": "工具返回结果"
    },
    {
      "from": "gpt",
      "value": "最终总结"
    }
  ],
  "system": "系统提示（可选，会被增强提示替换）",
  "tools": "工具定义JSON字符串"
}
```

## 训练流程

1. **数据验证**：检查数据是否符合约束
2. **数据增强**：应用增强的系统提示
3. **模型训练**：使用优化的超参数训练
4. **模型评估**：在测试集上评估性能

## 预期效果

通过本方案训练后，模型应该能够：

✅ 始终在检索阶段调用 `retrieval_tool`  
✅ 从返回的 top5 工具中选择合适的业务工具  
✅ 严格按照 `inputSchema` 提取参数  
✅ 生成准确完整的总结  
✅ 减少预期外的预测  

## 故障排除

### 问题1: 数据验证失败

**症状**：`validate_tool_calling_data.py` 报告错误

**解决**：
- 检查数据格式是否正确
- 确保所有样本都包含 `retrieval_tool` 调用
- 使用 `enhance_dataset_with_constraints.py` 修复数据

### 问题2: 训练时内存不足

**症状**：OOM (Out of Memory) 错误

**解决**：
- 减小 `per_device_train_batch_size`（当前为1）
- 减小 `gradient_accumulation_steps`（当前为16）
- 减小 `cutoff_len`（当前为8192）
- 启用 `gradient_checkpointing`（已启用）

### 问题3: 模型不遵循约束

**症状**：模型跳过 `retrieval_tool` 或调用错误的工具

**解决**：
- 检查增强的系统提示是否正确应用
- 增加训练轮数（当前为8.0）
- 检查训练数据质量
- 调整学习率（当前为2.0e-5）

## 高级用法

### 自定义系统提示

编辑 `enhanced_system_prompt.txt`，然后重新运行数据增强：

```bash
python enhance_dataset_with_constraints.py \
    data/dataset/12_08/train.json \
    data/dataset/12_08/train_enhanced.json
```

### 调整超参数

编辑 `run_enhanced_training_complete.py` 中的 `create_training_command` 函数，修改超参数值。

### 多GPU训练

使用 DeepSpeed 或 DDP：

```bash
# DDP (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    run_enhanced_training_complete.py
```

## 最佳实践

1. **数据质量**：确保训练数据质量高，符合约束要求
2. **数据增强**：使用增强的系统提示提高训练效果
3. **验证数据**：训练前验证数据，避免训练过程中的问题
4. **监控训练**：关注训练loss和验证指标
5. **迭代优化**：根据评估结果调整超参数和约束

## 联系与支持

如有问题或建议，请：
1. 检查本文档的故障排除部分
2. 查看训练日志
3. 验证数据格式和约束

## 更新日志

- **2025-12-08**: 初始版本
  - 创建增强的系统提示模板
  - 优化超参数配置
  - 实现约束验证机制
  - 创建完整的训练流程

