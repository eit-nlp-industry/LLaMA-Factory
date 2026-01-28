# Token上下文窗口功能说明

## 功能概述

已成功实现Token上下文窗口（Context Window）功能，将简单的"token loss高"升级为"token在特定上下文中loss高"，提供更精确的诊断信息。

## 实现内容

### 1. 数据结构增强

每个token记录现在包含完整的上下文信息：

```json
{
  "step": 100,
  "sample_id": 0,
  "position": 42,
  "gt_token_id": 321,
  "gt_token": ">",
  "gt_token_loss": 2.31,
  "token_type": "structural",
  "is_correct": false,
  "top1_pred_token": "}",
  "top1_pred_prob": 0.41,
  "topk_predictions": [
    {"token_id": 1257, "token": "}", "prob": 0.41},
    {"token_id": 321, "token": ">", "prob": 0.38},
    {"token_id": 1256, "token": "</anchor>", "prob": 0.09}
  ],
  "window_size": 10,
  "left_context_tokens": ["value", "=", "\"process", ".", "execute", "("],
  "right_context_tokens": ["\"", "\n", "<", "/", "anchor", ">"],
  "left_context_losses": [0.4, 0.2, 0.5, 0.3, 0.6, 0.4],
  "right_context_losses": [0.5, 0.6, 1.8, 2.1, 2.3, 2.0]
}
```

### 2. 新增字段说明

- **`gt_token_id`**: Ground truth token ID（原`token_id`）
- **`gt_token`**: Ground truth token文本（原`token`）
- **`gt_token_loss`**: Ground truth token的loss值（原`loss`）
- **`top1_pred_token`**: Top-1预测的token文本
- **`top1_pred_prob`**: Top-1预测的概率
- **`window_size`**: 上下文窗口大小（左右各包含的token数量，默认10）
- **`left_context_tokens`**: 左侧上下文tokens列表
- **`right_context_tokens`**: 右侧上下文tokens列表
- **`left_context_losses`**: 左侧上下文tokens的loss值列表
- **`right_context_losses`**: 右侧上下文tokens的loss值列表

### 3. 实现位置

**核心实现**：
- `src/llamafactory/train/token_loss_tracker.py`
  - `_extract_context_window()`: 提取上下文窗口的核心方法
  - `record_token_losses()`: 增强的记录方法，包含上下文提取

**配置**：
- `src/llamafactory/train/sft/trainer.py`
  - `TokenLossTracker`初始化时设置`context_window_size=10`

### 4. 上下文提取逻辑

#### 位置映射
- `pos`是在`shift_labels`中的位置（0到seq_len-2）
- `shift_labels[pos]`对应`input_ids[pos+1]`（因为labels是shifted的）
- 所以中心位置在input_ids中是`pos+1`

#### 左上下文
- 范围：`input_ids[pos+1-window_size : pos+1]`
- 包含当前token之前的所有上下文tokens
- 对应的losses：`sample_losses[pos-window_size : pos]`

#### 右上下文
- 范围：`input_ids[pos+2 : pos+2+window_size]`
- 包含当前token之后的所有上下文tokens
- 对应的losses：`sample_losses[pos+1 : pos+1+window_size]`

### 5. 分析脚本增强

**新增分析脚本**：
- `scripts/analyze_token_loss_with_context.py`: 专门的上下文分析脚本

**功能**：
1. **上下文模式分析**：识别token在特定上下文模式中的loss
2. **上下文loss分布分析**：分析特定token在不同上下文中的loss分布
3. **上下文可视化**：生成左右上下文loss分布直方图

**原有分析脚本更新**：
- `scripts/analyze_token_loss.py`: 已更新支持新旧字段名兼容

## 使用示例

### 训练时（自动）

训练时会自动记录上下文窗口，无需额外配置：

```bash
llamafactory-cli train ...
```

数据会自动保存到：`{output_dir}/token_loss_data/token_losses_step_*.jsonl`

### 训练后分析

#### 基础分析（支持上下文字段）

```bash
python scripts/analyze_token_loss.py \
    --token_loss_dir /path/to/token_loss_data \
    --output_dir ./analysis_results
```

#### 上下文专项分析

```bash
python scripts/analyze_token_loss_with_context.py \
    --token_loss_dir /path/to/token_loss_data \
    --output_dir ./context_analysis \
    --target_token "}"
```

## 诊断能力提升

### 之前（无上下文）

```
问题：token "}" 的loss高 (2.31)
结论：模型不懂 "}"
```

### 现在（有上下文）

```
问题：token "}" 的loss高 (2.31)
上下文：
  左侧: ["value", "=", "\"process", ".", "execute", "("]
  右侧: ["\"", "\n", "<", "/", "anchor", ">"]
  右侧loss: [0.5, 0.6, 1.8, 2.1, 2.3, 2.0]  ← 发现右侧loss也在升高

结论："}" 出现在 anchor 结构闭合位置时 loss 很高
     可能是模型在结构边界处容易出错
```

## 实际应用

### 识别结构边界问题

通过上下文分析，可以识别：
- 结构符号在特定结构中的loss模式
- 例如：`}`在`anchor`结构闭合时loss高，但在其他结构中正常

### 识别上下文依赖问题

- 某些token只在特定上下文中loss高
- 例如：`insert_after`关键字在特定XML结构中loss高

### 优化训练策略

根据上下文分析结果：
1. **数据增强**：增加包含问题上下文的训练样本
2. **Loss加权**：对特定上下文中的token给予更高权重
3. **Curriculum Learning**：先训练简单上下文，再逐步增加复杂上下文

## 性能考虑

- 上下文窗口大小默认10，可通过`context_window_size`参数调整
- 上下文提取使用numpy操作，性能开销很小
- 数据存储增加约20-30%（取决于窗口大小）

## 兼容性

- 分析脚本支持新旧字段名兼容
- 旧数据（无上下文字段）仍可正常分析
- 新数据（有上下文字段）可进行更深入的分析

## 下一步

1. **模式识别**：自动识别常见的问题上下文模式
2. **上下文聚类**：将相似的上下文模式聚类分析
3. **可视化增强**：生成上下文loss热力图

## 参考

- `TOKEN_LOSS_ANALYSIS_README.md`: 完整使用指南
- `src/llamafactory/train/token_loss_tracker.py`: 核心实现
- `scripts/analyze_token_loss_with_context.py`: 上下文分析脚本
