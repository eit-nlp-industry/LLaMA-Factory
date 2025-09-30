# LLaMA-Factory Function Call训练完整指南

## 1. 同步上游和本地更新代码

### 1.1 同步上游代码
```bash
# 进入LLaMA-Factory目录
cd LLaMA-Factory

# 添加上游仓库（如果还没有添加）
git remote add upstream https://github.com/hiyouga/LLaMA-Factory.git

# 获取上游最新代码
git fetch upstream

# 合并上游代码到本地分支
git merge upstream/main
```

### 1.2 处理冲突
如果出现冲突，需要手动解决冲突后提交：
```bash
# 解决冲突后
git add .
git commit -m "Merge upstream changes"
```

## 2. Dataset使用说明

### 2.1 数据组织结构
自定义数据全部放在 `LLaMA-Factory/data/dataset` 目录下，按照训练日期分割：
```
LLaMA-Factory/data/dataset/
├── 8_29/          # 8月29日的数据
├── 8_30/          # 8月30日的数据
├── 9_10/          # 9月10日的数据
│   └── function_call_data/  # function call数据目录
├── 9_14/          # 9月14日的数据
└── preprocess_dara/  # 数据预处理脚本目录
```

### 2.2 Function Call数据拼接
分散的function_call数据拼接成为多轮对话格式：

**数据位置**：`LLaMA-Factory/data/dataset/9_10/function_call_data/` 中的json数据

**拼接脚本**：`LLaMA-Factory/data/dataset/preprocess_dara/convert_to_function_call_format.py`

**重要提示**：
- 拼接的对话格式中，**偶数轮的信息才能被学习到，奇数轮的信息不能被学习到**
- 在构造hardmatch数据时需要特别注意这个规则
 
**使用示例**：
```bash
cd /home/ziqiang/LLaMA-Factory/data/dataset/preprocess_dara
python convert_to_function_call_format.py \
    --input_dir ../9_10/raw_data \
    --output_dir ../9_10/function_call_data \
    --output_file function_call_train.json
```

### 2.3 数据合并和格式转换

#### 2.3.1 混合训练数据合并
由于MCP function call使用多轮对话训练集结构，而价格服务部分使用单轮instruction input output格式，混训时需要统一格式。

**脚本**：`LLaMA-Factory/data/dataset/preprocess_dara/merge_and_convert_data.py`

**功能**：
- 将不同格式的数据转换为统一的sharegpt格式
- 自动在dataset_info.json中创建相应条目
- 支持多种数据格式的混合训练

**使用示例**：
```bash
python merge_and_convert_data.py \
    --function_call_data ../9_10/function_call_data/function_call_train.json \
    --price_service_data ../9_10/price_service_data.json \
    --output_file ../9_14/mixed_training_data.json \
    --dataset_name mixed_training_data
```

#### 2.3.2 简单JSON文件合并
如果只需要合并不同的JSON文件：

**脚本**：`LLaMA-Factory/data/dataset/preprocess_dara/merge_json_files.py`

**使用示例**：
```bash
python merge_json_files.py \
    --input_files file1.json file2.json file3.json \
    --output_file merged_data.json
```

### 2.4 创建dataset_info.json条目
每次新建dataset进行训练时，需要在 `LLaMA-Factory/data/dataset_info.json` 中创建相应条目。

**自动创建**：`merge_and_convert_data.py` 脚本目前支持自动添加条目

**手动创建示例**：
```json
{
  "mixed_training_data": {
    "file_name": "/home/ziqiang/LLaMA-Factory/data/dataset/9_14/mixed_training_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    }
  }
}
```

## 3. 训练和评估命令

### 3.1 训练命令

#### 3.1.1 基础训练命令
```bash
CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mixed_training_data_09_17 \
    --cutoff_len 8192 \
    --learning_rate 5e-05 \
    --num_train_epochs 5 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --output_dir /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_$(date +%Y-%m-%d-%H-%M) \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
```

#### 3.1.2 带调试日志的训练命令

```bash
# 将训练日志同时输出到控制台和文件
CUDA_VISIBLE_DEVICES=6,7 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mixed_training_data_09_17 \
    --cutoff_len 8192 \
    --learning_rate 5e-05 \
    --num_train_epochs 5 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --output_dir /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_$(date +%Y-%m-%d-%H-%M) \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target all \
    --gradient_checkpointing True \
    2>&1 | tee token_debug_current.log

CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mixed_training_data_09_17 \
    --cutoff_len 8192 \
    --learning_rate 5e-05 \
    --num_train_epochs 1 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --output_dir /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_$(date +%Y-%m-%d-%H-%M) \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --lora_target all \
    --gradient_checkpointing True \
    2>&1 | tee token_debug_current.log

```

#### 3.1.3 调试Token限制问题的训练命令
```bash
# 使用较小的cutoff_len来观察截断情况
CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mixed_training_data_09_17 \
    --cutoff_len 8192 \
    --learning_rate 5e-05 \
    --num_train_epochs 5 \
    --max_samples 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking False \
    --output_dir /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/debug_$(date +%Y-%m-%d-%H-%M) \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all 
```

**重要参数说明**：
- `--dataset`：指定在dataset_info.json中注册的条目名称
- `--output_dir`：LoRA模型保存位置
- `--enable_thinking`：是否训练模型的think模型
- `--cutoff_len`：Token长度限制，调试时建议使用2048观察截断情况
- `--max_samples`：调试时建议使用较小值（如1000）快速验证
- `2>&1 | tee`：将标准输出和错误输出同时显示在控制台和保存到文件

### 3.2 评估命令
```bash
CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft \
    --do_predict True \
    --model_name_or_path /data/models/Qwen3-8B \
    --adapter_name_or_path /home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_2025-09-14-10-48-context/checkpoint-430 \
    --preprocessing_num_workers 8 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset test_data \
    --cutoff_len 2048 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --max_new_tokens 1024 \
    --do_sample False \
    --temperature 0.0 \
    --top_p 1.0 \
    --bf16 True \
    --trust_remote_code True \
    --output_dir Qwen3-8B/eval_results/9_14
```

**重要参数说明**：
- `--eval_dataset`：指定在dataset_info.json中注册的测试数据集名称
- `--output_dir`：评估结果保存位置

### 3.3 评估结果分析
评估后会生成 `generated_predictions.jsonl` 文件，包含每条数据的label和predict。

**价格服务评估脚本**：`LLaMA-Factory/data/dataset/preprocess_dara/eval_by_field.py`

**使用示例**：
```bash
python eval_by_field.py \
    --predictions_file Qwen3-8B/eval_results/9_14/generated_predictions.jsonl \
    --output_file eval_results_by_field.json
```

## 4. Function Call训练数据的Loss Mask机制详解

### 4.1 数据转换过程

#### 原始对话数据：
```json
{
  "conversations": [
    {"from": "system", "value": "你是一个智能助手..."},
    {"from": "human", "value": "报表编号H20250611的收入类型分布情况能分析一下吗？"},
    {"from": "function_call", "value": "{\"name\": \"analyze_revenue_by_type\", \"arguments\": {...}}"},
    {"from": "observation", "value": "工具返回的结果..."},
    {"from": "gpt", "value": "<answer>根据分析结果...</answer>"}
  ]
}
```

#### 转换为Token序列：
```
<|im_start|>system
你是一个智能助手...
<|im_end|>
<|im_start|>user
报表编号H20250611的收入类型分布情况能分析一下吗？
<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "analyze_revenue_by_type", "arguments": {...}}
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
工具返回的结果...
</tool_response>
<|im_end|>
<|im_start|>assistant
<answer>根据分析结果...</answer>
<|im_end|>
```

### 4.2 Loss Mask应用

#### Input IDs (完整序列):
```
[系统消息tokens] [用户查询tokens] [工具调用tokens] [工具结果tokens] [最终回答tokens]
```

#### Labels (用于计算loss):
```
[   IGNORE_INDEX   ] [  IGNORE_INDEX  ] [工具调用tokens] [  IGNORE_INDEX  ] [最终回答tokens]
     (不训练)           (不训练)            (训练)          (不训练)           (训练)
```

### 4.3 训练目标

模型学习的目标是：
1. **根据用户查询生成正确的工具调用** - 从用户输入预测function_call部分
2. **根据工具结果生成合适的回答** - 从工具返回结果预测最终答案

模型**不会**学习：
1. 生成工具返回结果（observation部分被mask）
2. 重复用户输入或系统提示

### 4.4 关键代码实现

#### Template中的格式定义：
```python
# Qwen3模板
format_observation=StringFormatter(
    slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
)
```

#### Loss计算时的mask：
```python
# 在supervised.py中
if self.data_args.train_on_prompt:
    source_label = source_ids
else:
    source_label = [IGNORE_INDEX] * source_len  # 输入部分被mask

# observation部分会被自动识别并mask为IGNORE_INDEX
```

## 5. Token调试日志系统

### 5.1 概述

为了帮助开发者更好地理解和调试ShareGPT格式训练中的token限制问题，我们在关键代码位置添加了详细的调试日志系统。

### 5.2 调试日志类型

#### 5.2.1 [TEMPLATE_DEBUG] - 模板编码阶段
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

#### 5.2.2 [TOKEN_DEBUG] - 数据编码阶段
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

#### 5.2.3 [INFER_SEQLEN] - 截断策略阶段
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

### 5.3 日志查看方法

#### 5.3.1 运行训练并记录日志
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

#### 5.3.2 过滤特定日志
```bash
# 查看所有token调试日志
grep 'TOKEN_DEBUG' debug_train.log

# 查看截断策略日志
grep 'INFER_SEQLEN' debug_train.log

# 查看模板编码日志
grep 'TEMPLATE_DEBUG' debug_train.log

# 查看截断事件
grep '⚠️' debug_train.log

# 查看使用率统计
grep '使用率:' debug_train.log
```

#### 5.3.3 使用分析脚本
```bash
# 分析日志文件
python analyze_token_logs.py debug_train.log
```

### 5.4 关键指标解读

#### 5.4.1 使用率 (Usage Rate)
- **100%**: 完全使用cutoff_len，可能有截断
- **<100%**: 未完全使用，数据较短
- **>100%**: 不可能出现，检查日志

#### 5.4.2 截断事件
- **source被截断**: 通常是observation内容过长
- **target被截断**: 通常是assistant回复被截断
- **预算耗尽**: 后续pairs被完全丢弃

#### 5.4.3 截断策略
- **策略1**: target完全保留，截断source (target_len * 2 < cutoff_len)
- **策略2**: source完全保留，截断target (source_len * 2 < cutoff_len)  
- **策略3**: 按比例截断source和target

### 5.5 优化建议

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

### 5.6 日志文件位置

训练过程中的日志会输出到以下位置：

1. **统一调试日志文件**: `token_debug_YYYYMMDD_HHMMSS.log`（自动生成时间戳）
   - 包含所有 `[TOKEN_DEBUG]`、`[TEMPLATE_DEBUG]`、`[INFER_SEQLEN]` 日志
   - 自动轮转：文件大小达到50MB时自动创建新文件
   - 自动清理：保留3天的历史日志
   - 异步写入：不影响训练性能

2. **控制台输出**: 直接显示在终端（彩色格式）
3. **重定向文件**: 如果使用 `tee` 命令，会保存到指定文件
4. **训练日志**: 在 `output_dir` 目录下的 `trainer_log.jsonl` 文件
5. **系统日志**: 根据系统配置，可能输出到 `/var/log/` 或其他位置

**推荐做法**:
```bash
# 将日志同时输出到控制台和文件
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train [参数] 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

# 或者只保存到文件
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train [参数] > training.log 2>&1
```

**查看统一调试日志**:
```bash
# 实时查看调试日志
tail -f token_debug_*.log

# 过滤特定类型的调试信息
grep "TOKEN_DEBUG" token_debug_*.log
grep "TEMPLATE_DEBUG" token_debug_*.log  
grep "INFER_SEQLEN" token_debug_*.log

# 查看截断事件
grep "截断" token_debug_*.log

# 统计使用率
grep "使用率" token_debug_*.log | tail -10
```

### 5.7 注意事项

1. **性能影响**: 调试日志会增加训练时间，建议只在调试时使用
2. **日志量**: 日志量较大，建议重定向到文件
3. **生产环境**: 生产环境建议移除或注释掉调试日志
4. **存储空间**: 长时间训练会产生大量日志，注意磁盘空间

## 6. 快速参考

### 6.1 常用调试命令

```bash
# 快速调试token截断问题
CUDA_VISIBLE_DEVICES=6 llamafactory-cli train \
    --stage sft --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --dataset your_dataset --cutoff_len 2048 \
    --max_samples 100 --num_train_epochs 1 \
    --output_dir ./debug_output \
    2>&1 | tee debug.log

# 查看截断事件
grep '⚠️' debug.log

# 查看使用率统计
grep '使用率:' debug.log

# 分析日志
python analyze_token_logs.py debug.log
```

### 6.2 日志文件位置总结

| 日志类型 | 位置 | 说明 |
|---------|------|------|
| **统一调试日志** | `token_debug_*.log` | 包含所有`[TOKEN_DEBUG]`等标记的调试信息 |
| **控制台输出** | 终端 | 实时显示，训练时可见 |
| **重定向文件** | `training_*.log` | 使用`tee`命令保存的完整日志 |
| **训练日志** | `output_dir/trainer_log.jsonl` | LLaMA-Factory生成的训练日志 |

### 6.3 关键文件路径

```
LLaMA-Factory/
├── src/llamafactory/data/
│   ├── processor/supervised.py          # TOKEN_DEBUG日志
│   ├── processor/processor_utils.py     # INFER_SEQLEN日志
│   └── template.py                      # TEMPLATE_DEBUG日志
├── configure_token_logs.py              # 日志配置脚本
├── token_debug_*.log                    # 统一调试日志文件
├── test_token_debug.py                  # 测试脚本
├── analyze_token_logs.py                # 日志分析脚本
└── TOKEN_DEBUG_README.md                # 快速使用指南
```

### 6.4 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 日志不显示 | 日志级别设置 | 检查logging配置 |
| 截断频繁 | cutoff_len太小 | 增加cutoff_len到4096或8192 |
| 使用率低 | 数据过短 | 考虑减少cutoff_len |
| 训练效果差 | assistant被截断 | 优先解决截断问题 |

