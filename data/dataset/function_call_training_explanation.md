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
```bash
CUDA_VISIBLE_DEVICES=2,6 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset mixed_training_data \
    --cutoff_len 2048 \
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
    --report_to none \
    --output_dir saves/Qwen3-8B/lora/train_2025-09-14-10-48-context \
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

