# Token调试日志使用指南

## 概述

为了帮助开发者更好地理解和调试LLaMA-Factory中ShareGPT格式训练的token限制问题，我们在关键代码位置添加了详细的调试日志。

## 添加的调试日志

### 1. [TEMPLATE_DEBUG] - 模板编码阶段
**位置**: `src/llamafactory/data/template.py` 的 `encode_multiturn` 函数

**记录内容**:
- 输入messages数量
- 编码后messages数量  
- 生成的pairs数量和每个pair的token长度

**示例输出**:
```
[TEMPLATE_DEBUG] encode_multiturn开始
[TEMPLATE_DEBUG] 输入messages数量: 6
[TEMPLATE_DEBUG] 编码后messages数量: 6
[TEMPLATE_DEBUG] 生成的pairs数量: 3
[TEMPLATE_DEBUG] Pair 1: source=120 tokens, target=80 tokens
[TEMPLATE_DEBUG] Pair 2: source=1300 tokens, target=50 tokens
[TEMPLATE_DEBUG] Pair 3: source=400 tokens, target=200 tokens
```

### 2. [TOKEN_DEBUG] - 数据编码阶段
**位置**: `src/llamafactory/data/processor/supervised.py` 的 `_encode_data_example` 函数

**记录内容**:
- 原始conversations长度
- 编码后的pairs数量
- 每个pair的原始长度和截断后长度
- 剩余预算和累计长度
- 最终结果和使用率

**示例输出**:
```
[TOKEN_DEBUG] 开始处理数据样本
[TOKEN_DEBUG] 原始conversations长度: 6 条消息
[TOKEN_DEBUG] 编码后的pairs数量: 3
[TOKEN_DEBUG] 初始total_length: 0
[TOKEN_DEBUG] cutoff_len: 2048
[TOKEN_DEBUG] === Pair 1 ===
[TOKEN_DEBUG] 原始长度: source=120, target=80
[TOKEN_DEBUG] 剩余预算: 2048
[TOKEN_DEBUG] 截断后长度: source=120->120, target=80->80
[TOKEN_DEBUG] 当前累计长度: 200/2048
[TOKEN_DEBUG] === Pair 2 ===
[TOKEN_DEBUG] 原始长度: source=1300, target=50
[TOKEN_DEBUG] 剩余预算: 1848
[TOKEN_DEBUG] 截断后长度: source=1300->1300, target=50->50
[TOKEN_DEBUG] 当前累计长度: 1550/2048
[TOKEN_DEBUG] === Pair 3 ===
[TOKEN_DEBUG] 原始长度: source=400, target=200
[TOKEN_DEBUG] 剩余预算: 498
[TOKEN_DEBUG] 截断后长度: source=400->298, target=200->200
[TOKEN_DEBUG] ⚠️ source被截断: 102 tokens
[TOKEN_DEBUG] 当前累计长度: 2048/2048
[TOKEN_DEBUG] === 最终结果 ===
[TOKEN_DEBUG] 最终input_ids长度: 2048
[TOKEN_DEBUG] 最终labels长度: 2048
[TOKEN_DEBUG] 最终total_length: 2048
[TOKEN_DEBUG] 使用率: 2048/2048 (100.0%)
[TOKEN_DEBUG] 处理完成
```

### 3. [INFER_SEQLEN] - 截断策略阶段
**位置**: `src/llamafactory/data/processor/processor_utils.py` 的 `infer_seqlen` 函数

**记录内容**:
- 截断策略的选择过程
- 输入输出参数
- 具体的截断逻辑

**示例输出**:
```
[INFER_SEQLEN] 输入: source_len=400, target_len=200, cutoff_len=498
[INFER_SEQLEN] 条件1: target_len*2 < cutoff_len (200*2=400 < 498)
[INFER_SEQLEN] 策略1: target完全保留，截断source
[INFER_SEQLEN] 输出: source_len=400->298, target_len=200->200
```

## 使用方法

### 1. 运行训练并记录日志

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --dataset your_dataset \
    --cutoff_len 2048 \
    --output_dir ./debug_output \
    2>&1 | tee debug_train.log
```

### 2. 过滤特定日志

```bash
# 查看所有token调试日志
grep 'TOKEN_DEBUG' debug_train.log

# 查看截断策略日志
grep 'INFER_SEQLEN' debug_train.log

# 查看模板编码日志
grep 'TEMPLATE_DEBUG' debug_train.log

# 查看截断事件
grep '⚠️' debug_train.log
```

### 3. 使用分析脚本

```bash
# 分析日志文件
python analyze_token_logs.py debug_train.log
```

分析脚本会输出:
- 总样本数和截断统计
- Token使用率分布
- 截断事件分析
- 使用率分布统计

## 关键指标解读

### 1. 使用率 (Usage Rate)
- **100%**: 完全使用cutoff_len，可能有截断
- **<100%**: 未完全使用，数据较短
- **>100%**: 不可能出现，检查日志

### 2. 截断事件
- **source被截断**: 通常是observation内容过长
- **target被截断**: 通常是assistant回复被截断
- **预算耗尽**: 后续pairs被完全丢弃

### 3. 截断策略
- **策略1**: target完全保留，截断source (target_len * 2 < cutoff_len)
- **策略2**: source完全保留，截断target (source_len * 2 < cutoff_len)  
- **策略3**: 按比例截断source和target

## 优化建议

基于日志分析结果:

1. **如果observation经常被截断**:
   - 增加cutoff_len到4096或8192
   - 压缩observation内容长度

2. **如果assistant回复被截断**:
   - 这是最严重的问题，必须解决
   - 优先增加cutoff_len

3. **如果使用率很低**:
   - 考虑减少cutoff_len以提高训练效率

4. **如果经常预算耗尽**:
   - 数据过长，需要预处理压缩

## 注意事项

1. 调试日志会增加训练时间，建议只在调试时使用
2. 日志量较大，建议重定向到文件
3. 可以通过修改日志级别来控制输出量
4. 生产环境建议移除或注释掉调试日志

## 文件修改列表

- `src/llamafactory/data/processor/supervised.py`: 添加TOKEN_DEBUG日志
- `src/llamafactory/data/processor/processor_utils.py`: 添加INFER_SEQLEN日志  
- `src/llamafactory/data/template.py`: 添加TEMPLATE_DEBUG日志
- `test_token_debug.py`: 测试脚本
- `analyze_token_logs.py`: 日志分析脚本
- `TOKEN_DEBUG_GUIDE.md`: 本使用指南
