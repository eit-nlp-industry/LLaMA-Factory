# Token-level Loss 分析指南

本指南说明如何使用LLaMA-Factory的token-level loss跟踪和分析功能，深入分析SFT训练中哪些token的预测结果总是有问题。

## 功能概述

该功能实现了以下四个关键分析维度：

1. **Token-level Loss 直接打点**：在训练过程中记录每个token的loss值
2. **高频高Loss Token统计**：识别出现频率高且loss高的token
3. **Token类型聚类分析**：按token类型（结构符号、关键字、数值、路径、自然语言）统计平均loss
4. **位置敏感分析**：分析不同序列位置的loss分布
5. **Top-k预测对比**：分析模型预测的top-k结果，识别低置信度但正确的预测

## 使用方法

### 1. 训练时自动记录Token-level Loss

在训练过程中，系统会自动记录token-level loss数据。无需额外配置，训练时会自动：

- 在每次forward pass时计算per-token loss（使用`reduction="none"`）
- 记录token信息、loss值、top-k预测结果
- 定期保存到 `{output_dir}/token_loss_data/` 目录

**数据格式**：
每个记录包含：
```json
{
  "step": 100,
  "sample_id": 0,
  "position": 42,
  "gt_token_id": 1256,
  "gt_token": "}",
  "gt_token_loss": 2.31,
  "token_type": "structural",
  "is_correct": false,
  "top1_pred_token": "}",
  "top1_pred_prob": 0.41,
  "topk_predictions": [
    {"token_id": 1257, "token": "}", "prob": 0.41},
    {"token_id": 1256, "token": "}", "prob": 0.38},
    ...
  ],
  "window_size": 10,
  "left_context_tokens": ["value", "=", "\"process", ".", "execute", "("],
  "right_context_tokens": ["\"", "\n", "<", "/", "anchor", ">"],
  "left_context_losses": [0.4, 0.2, 0.5, 0.3, 0.6, 0.4],
  "right_context_losses": [0.5, 0.6, 1.8, 2.1, 2.3, 2.0]
}
```

**上下文窗口说明**：
- `window_size`: 左右各包含的token数量（默认10）
- `left_context_tokens`: 当前token左侧的上下文tokens
- `right_context_tokens`: 当前token右侧的上下文tokens
- `left_context_losses`: 左侧上下文tokens的loss值
- `right_context_losses`: 右侧上下文tokens的loss值

这样可以将"`}`的loss高"升级为"`}`出现在anchor结构闭合位置时loss很高"，提供更精确的诊断信息。

### 2. 训练后分析数据

训练完成后，使用分析脚本分析收集的数据：

```bash
python scripts/analyze_token_loss.py \
    --token_loss_dir /path/to/training/output/token_loss_data \
    --output_dir ./analysis_results
```

**输出文件**：
- `high_loss_tokens.csv`: 高频高loss token统计
- `token_type_analysis.csv`: 按token类型统计
- `position_analysis.csv`: 按位置统计
- `topk_prediction_analysis.csv`: Top-k预测分析
- `position_loss_analysis.png`: 位置loss可视化图
- `token_loss_analysis_report.md`: 综合分析报告

## 分析结果解读

### 1. 高频高Loss Token统计

这个分析会显示：
- 哪些token出现频率高且loss高
- 这些token的平均loss、标准差、最大最小loss

**示例输出**：
```
Top 50 高频高Loss Tokens:
token        token_id  token_type    count  avg_loss  std_loss
</xml>       1256      structural    150    2.8       0.5
}            1257      structural    200    2.4       0.6
,            44        structural    500    2.1       0.4
```

**解读**：
- 如果看到大量结构符号（`}`, `</xml>`, `,`）出现在高loss列表中，说明模型在结构边界处容易出错
- 如果看到特定关键字（如`insert_after`）loss高，说明这些domain-specific token需要更多训练

### 2. Token类型聚类分析

按token类型统计平均loss，帮助识别哪类token是训练瓶颈。

**示例输出**：
```
token_type        count  avg_loss
structural        5000   2.5
keyword           2000   1.9
numeric           1000   1.2
natural_language  10000  1.0
```

**解读**：
- 如果`structural`类型的loss明显高于其他类型，说明模型在结构符号预测上有困难
- 如果`natural_language`的loss最低，说明模型对自然语言部分学得较好

### 3. 位置敏感分析

分析不同序列位置的loss分布，识别问题位置。

**关键发现**：
- **位置0-50**: 通常是instruction部分，loss应该较低
- **位置50-150**: 通常是output开始部分，loss应该稳定
- **位置150+**: 如果loss飙升，可能是：
  - Context太长导致注意力分散
  - Generation尾部崩溃（常见问题）
  - Instruction和output边界没学稳

**可视化**：
`position_loss_analysis.png` 会显示loss随位置的变化曲线，红色高亮高loss区域。

### 4. Top-k预测对比分析

分析模型预测的置信度和正确性。

**关键指标**：
- **Top-1正确率**: 模型直接预测正确的比例
- **Top-5包含正确答案**: 正确答案在top-5中的比例
- **高Loss但预测正确**: 这些token虽然预测对了，但模型不够confident

**解读**：
- 如果Top-1正确率低但Top-5包含率高，说明模型知道大概是什么，但在精确形式上不稳定
- 如果高loss但预测正确，说明这些token需要更多训练来提高置信度

## 实际应用建议

### 针对结构符号高Loss

如果发现结构符号（`}`, `</xml>`, `,`）loss高：

1. **数据增强**：增加包含这些符号的训练样本
2. **Loss加权**：对结构符号的loss给予更高权重
3. **Curriculum Learning**：先训练简单样本，再逐步增加复杂结构

### 针对位置尾部高Loss

如果发现序列尾部loss高：

1. **缩短Context**：减少输入长度
2. **增加尾部样本**：专门收集长序列的训练数据
3. **调整Attention Mask**：确保模型能关注到尾部

### 针对特定Token类型高Loss

如果发现某类token（如关键字）loss高：

1. **增加相关样本**：专门收集包含这些token的训练数据
2. **Fine-tune策略**：对这些token进行针对性训练
3. **数据平衡**：确保训练数据中各类token分布均衡

## 技术细节

### Token类型分类规则

- **structural**: 包含 `{`, `}`, `[`, `]`, `<`, `>`, `/`, `,`, `:`, `;`, `(`, `)`, `"`, `'` 或XML/HTML标签
- **keyword**: 常见编程/格式关键字（`insert_after`, `anchor`, `function`, `method`, `class`, `def`, `return`, `if`, `else`, `for`, `while`）
- **numeric**: 数字模式（纯数字、十六进制如`0x...`）
- **path**: 路径模式（包含`/`或`\`，或`.xml`、`.json`后缀）
- **natural_language**: 其他所有token（默认）

### 性能考虑

- Token-level loss记录使用`detach()`避免影响梯度计算
- 默认每100条记录保存一次，减少I/O开销
- 每个训练步骤只记录1个样本（`max_samples_per_step=1`），可通过修改`TokenLossTracker`初始化参数调整

## 故障排除

### 问题：没有生成token_loss_data目录

**解决**：检查训练是否正常启动，确保`CustomSeq2SeqTrainer`被正确使用。

### 问题：分析脚本找不到数据文件

**解决**：确保`--token_loss_dir`指向正确的目录，应该包含`token_losses_step_*.jsonl`文件。

### 问题：内存不足

**解决**：
1. 减少`max_samples_per_step`（默认1）
2. 增加`save_interval`（默认100），更频繁地保存和清空内存
3. 只分析部分数据文件

## 示例工作流

```bash
# 1. 开始训练（自动记录token-level loss）
llamafactory-cli train ...

# 2. 训练完成后，分析数据
python scripts/analyze_token_loss.py \
    --token_loss_dir ./saves/Qwen3-8B/lora/train_2026-01-12-11-20/token_loss_data \
    --output_dir ./analysis_results

# 3. 查看分析结果
cat analysis_results/token_loss_analysis_report.md
open analysis_results/position_loss_analysis.png
```

## 参考

- 原始需求文档中的分析思路
- LLaMA-Factory训练流程文档
- Transformers Trainer文档
