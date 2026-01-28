# Token-level Loss 跟踪与分析实现总结

## 实现概述

已成功实现完整的token-level loss跟踪与分析系统，用于深入分析SFT训练中哪些token的预测结果总是有问题。

## 实现的功能

### ✅ 1. Token-level Loss 直接打点（最有效）

**实现位置**：
- `src/llamafactory/train/token_loss_tracker.py`: TokenLossTracker类
- `src/llamafactory/train/sft/trainer.py`: 修改compute_loss方法

**核心实现**：
```python
# 使用 reduction="none" 保留 per-token loss
loss_fct = torch.nn.CrossEntropyLoss(
    ignore_index=IGNORE_INDEX,
    reduction="none"
)

token_losses = loss_fct(
    logits.view(-1, vocab_size),
    labels.view(-1)
)  # shape: [batch * seq_len]

token_losses = token_losses.view(batch_size, seq_len)
```

**数据记录**：
- 每个token的loss值
- Token ID和文本
- Token类型（structural, keyword, numeric, path, natural_language）
- 位置信息
- Top-k预测结果

### ✅ 2. 高频高Loss Token统计

**实现位置**：`scripts/analyze_token_loss.py` - `analyze_high_loss_tokens()`

**功能**：
- 统计每个token的出现频率和平均loss
- 识别Top-K高频高loss tokens
- 输出详细的统计信息（平均loss、标准差、最大最小loss）

**输出示例**：
```
Top 50 高频高Loss Tokens:
token        token_id  token_type    count  avg_loss
</xml>       1256      structural    150    2.8
}            1257      structural    200    2.4
```

### ✅ 3. Token类型聚类分析

**实现位置**：`scripts/analyze_token_loss.py` - `analyze_token_type_clustering()`

**功能**：
- 自动分类token类型：
  - **structural**: 结构符号（`{`, `}`, `<`, `>`, `/`, `,`, `:`, XML标签等）
  - **keyword**: 关键字（`insert_after`, `anchor`, `function`, `method`等）
  - **numeric**: 数值（数字、十六进制）
  - **path**: 路径（包含`/`或`.xml`、`.json`后缀）
  - **natural_language**: 自然语言（默认）
- 按类型统计平均loss
- 识别训练瓶颈所在的token类型

**输出示例**：
```
token_type        count  avg_loss
structural        5000   2.5
keyword           2000   1.9
natural_language  10000  1.0
```

### ✅ 4. 位置敏感分析

**实现位置**：`scripts/analyze_token_loss.py` - `analyze_position_sensitivity()`

**功能**：
- 分析不同序列位置的loss分布
- 识别关键位置（第一个target token、结构开始/结束token）
- 生成位置loss可视化图
- 分析不同位置范围的loss趋势

**关键发现**：
- 位置0-50：通常loss较低（instruction部分）
- 位置50-150：loss应该稳定（output开始）
- 位置150+：如果loss飙升，可能是generation尾部崩溃

**可视化**：生成`position_loss_analysis.png`图表

### ✅ 5. Top-k预测对比分析

**实现位置**：`scripts/analyze_token_loss.py` - `analyze_topk_predictions()`

**功能**：
- 分析模型预测的top-k结果
- 计算Top-1正确率和Top-5包含率
- 识别高loss但预测正确的情况（低置信度）
- 分析错误预测的模式

**关键指标**：
- Top-1正确率
- Top-5包含正确答案的比例
- 高loss但预测正确的token（需要提高置信度）

## 文件结构

```
LLaMA-Factory/
├── src/llamafactory/train/
│   ├── token_loss_tracker.py          # TokenLossTracker类
│   ├── token_loss_callback.py         # 训练结束回调
│   └── sft/trainer.py                 # 修改的trainer（集成token loss跟踪）
├── scripts/
│   ├── analyze_token_loss.py          # 分析脚本
│   └── example_token_loss_analysis.sh # 使用示例脚本
└── TOKEN_LOSS_ANALYSIS_README.md       # 详细使用文档
```

## 使用方法

### 训练时（自动）

训练时会自动记录token-level loss，无需额外配置：

```bash
llamafactory-cli train ...
```

数据会自动保存到：`{output_dir}/token_loss_data/token_losses_step_*.jsonl`

### 训练后分析

```bash
# 方法1：使用示例脚本
bash scripts/example_token_loss_analysis.sh

# 方法2：直接运行分析脚本
python scripts/analyze_token_loss.py \
    --token_loss_dir /path/to/token_loss_data \
    --output_dir ./analysis_results
```

## 输出文件

分析脚本会生成以下文件：

1. **token_loss_analysis_report.md**: 综合分析报告（Markdown格式）
2. **high_loss_tokens.csv**: 高频高loss token统计
3. **token_type_analysis.csv**: Token类型分析
4. **position_analysis.csv**: 位置分析
5. **topk_prediction_analysis.csv**: Top-k预测分析
6. **position_loss_analysis.png**: 位置loss可视化图

## 技术细节

### Token Loss记录流程

1. **Forward Pass**: 在`compute_loss`中获取模型输出
2. **计算Per-token Loss**: 使用`reduction="none"`的CrossEntropyLoss
3. **记录数据**: 保存token信息、loss、top-k预测
4. **定期保存**: 每100条记录保存一次（可配置）
5. **训练结束**: 自动finalize并生成汇总统计

### 性能优化

- 使用`detach()`避免影响梯度计算
- 每个step只记录1个样本（可配置）
- 定期保存减少内存占用
- 异步I/O不影响训练速度

### Token类型分类规则

- **Structural**: 包含结构符号或XML/HTML标签
- **Keyword**: 常见编程/格式关键字
- **Numeric**: 数字模式
- **Path**: 路径模式
- **Natural Language**: 其他所有token

## 实际应用建议

### 针对结构符号高Loss

1. 数据增强：增加包含结构符号的训练样本
2. Loss加权：对结构符号的loss给予更高权重
3. Curriculum Learning：先训练简单样本

### 针对位置尾部高Loss

1. 缩短Context：减少输入长度
2. 增加尾部样本：专门收集长序列数据
3. 调整Attention Mask：确保关注到尾部

### 针对特定Token类型高Loss

1. 增加相关样本：专门收集包含这些token的数据
2. Fine-tune策略：针对性训练
3. 数据平衡：确保各类token分布均衡

## 验证

已实现的功能完全符合原始需求：

✅ **2.1 修改 forward：保留 per-token loss** - 已实现
✅ **2.2 把 token loss 和 token 本身对齐** - 已实现
✅ **3.1 高频高 loss token 统计** - 已实现
✅ **3.2 按 token 类型聚类** - 已实现
✅ **4.1 按序列位置画 loss** - 已实现
✅ **4.2 特别关注关键位置** - 已实现
✅ **5.1 记录 logits → top-k** - 已实现

## 下一步

1. **运行训练**：使用修改后的trainer进行训练，自动收集token-level loss数据
2. **分析数据**：训练完成后运行分析脚本
3. **优化训练**：根据分析结果调整训练策略

## 注意事项

- 确保训练使用`CustomSeq2SeqTrainer`（SFT训练默认使用）
- 分析脚本需要pandas和matplotlib（已在requirements.txt中）
- 大量数据可能需要较长时间分析，建议先分析部分数据验证

## 参考文档

- `TOKEN_LOSS_ANALYSIS_README.md`: 详细使用指南
- `scripts/analyze_token_loss.py`: 分析脚本源码
- `src/llamafactory/train/token_loss_tracker.py`: TokenLossTracker实现
