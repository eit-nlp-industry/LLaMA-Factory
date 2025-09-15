# Function Call训练数据的Loss Mask机制详解

## 1. 数据转换过程

### 原始对话数据：
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

### 转换为Token序列：
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

## 2. Loss Mask应用

### Input IDs (完整序列):
```
[系统消息tokens] [用户查询tokens] [工具调用tokens] [工具结果tokens] [最终回答tokens]
```

### Labels (用于计算loss):
```
[   IGNORE_INDEX   ] [  IGNORE_INDEX  ] [工具调用tokens] [  IGNORE_INDEX  ] [最终回答tokens]
     (不训练)           (不训练)            (训练)          (不训练)           (训练)
```

## 3. 训练目标

模型学习的目标是：
1. **根据用户查询生成正确的工具调用** - 从用户输入预测function_call部分
2. **根据工具结果生成合适的回答** - 从工具返回结果预测最终答案

模型**不会**学习：
1. 生成工具返回结果（observation部分被mask）
2. 重复用户输入或系统提示

## 4. 关键代码实现

### Template中的格式定义：
```python
# Qwen3模板
format_observation=StringFormatter(
    slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
)
```

### Loss计算时的mask：
```python
# 在supervised.py中
if self.data_args.train_on_prompt:
    source_label = source_ids
else:
    source_label = [IGNORE_INDEX] * source_len  # 输入部分被mask

# observation部分会被自动识别并mask为IGNORE_INDEX
```

## 5. 实际效果

这种设计确保了：
1. 模型学会**何时**调用工具
2. 模型学会**如何**调用工具（正确的参数）
3. 模型学会**处理**工具返回结果
4. 模型**不会**试图"预测"或"生成"工具的返回结果

这样训练出来的模型能够：
- 理解用户需求并选择合适的工具
- 构造正确的工具调用参数
- 基于真实的工具返回结果给出合理的回答
- 不会出现"幻觉"工具返回结果的问题
