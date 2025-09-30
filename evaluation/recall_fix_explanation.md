# Recall@5 评估问题修复说明

## 问题描述

在异步并发版本的评测脚本中，发现 `recall@5` 指标没有被正确评估。从日志可以看到：

```
2025-09-25 09:59:13.722 | INFO | Pair 2 评估完成，accuracy: 0.833, precision@1: 1.000
```

注意到日志中只显示了 `precision@1: 1.000`，但没有显示 `recall@5` 的值，这说明 `recall` 为 `None`。

## 问题根因

在异步并发处理中，**pair2可能比pair1先完成处理**，导致当pair2需要计算recall时，pair1的预测结果还不存在。

### 具体分析

1. **依赖关系**：Pair2的recall计算依赖于Pair1的预测结果
   ```python
   if pair.pair_id == 2 and not DISABLE_RECALL:
       pair1_predict = pair_predict_by_id.get(1)  # 这里可能为None
       pair1_toolname_score = pair_toolname_score_by_id.get(1)
       if pair1_predict and pair1_toolname_score == 1.0:
           recall, recall_details = await self.retrieval_caller.compute_recall_from_pair1_predict(...)
   ```

2. **并发执行问题**：原始异步实现中，所有pairs都是并发执行的
   ```python
   # 原始代码 - 所有pairs并发执行
   pair_results = await asyncio.gather(*pair_tasks, return_exceptions=True)
   ```

3. **时序问题**：当Pair2完成时，Pair1可能还没完成，导致 `pair_predict_by_id.get(1)` 返回 `None`

## 解决方案

修改异步处理逻辑，确保**tool_call类型的pairs按顺序处理**，而text_generation类型的pairs可以并发处理。

### 修复后的处理逻辑

```python
# 需要按顺序处理tool_call类型的pairs，因为pair2依赖pair1的结果
sorted_pairs = sorted(unprocessed_pairs, key=lambda p: p.pair_id)

results = []
text_gen_pairs = []

# 分离tool_call和text_generation类型的pairs
for pair in sorted_pairs:
    if pair.pair_type == "tool_call":
        # tool_call类型的pairs需要串行处理，确保依赖关系
        result = await self._evaluate_single_pair_async(...)
        results.append(result)
    else:
        # text_generation类型的pairs收集起来并发处理
        text_gen_pairs.append(pair)

# 并发处理所有text_generation类型的pairs
if text_gen_pairs:
    text_gen_results = await asyncio.gather(*text_gen_tasks, return_exceptions=True)
    # 处理结果...
```

### 关键改进

1. **按pair_id排序**：确保pair1在pair2之前处理
2. **串行处理tool_call**：确保依赖关系正确
3. **并发处理text_generation**：保持性能优势
4. **保持并发优势**：对话级别和text_generation级别的并发仍然有效

## 验证方法

修复后，应该能看到类似以下的日志：

```
2025-09-25 09:59:13.722 | INFO | Pair 2 评估完成，accuracy: 0.833, precision@1: 1.000, recall@5: 1
```

其中 `recall@5` 会显示具体的值（0或1）。

## 性能影响

- **轻微性能损失**：tool_call类型的pairs从并发改为串行
- **保持并发优势**：对话级别和text_generation类型的并发仍然有效
- **正确性优先**：确保评估结果的正确性

## 测试建议

1. 运行小规模测试验证recall@5指标正常显示
2. 对比修复前后的评估结果一致性
3. 监控性能影响是否在可接受范围内

## 总结

这个修复解决了异步并发版本中recall@5指标缺失的问题，通过合理的依赖关系处理，既保证了评估的正确性，又最大程度保持了并发处理的性能优势。
