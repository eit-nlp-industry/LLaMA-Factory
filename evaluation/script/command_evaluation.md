# 训练数据评估脚本使用说明

## 一、评估结构
解析conversations数据，分割成pairs
- **Pair 1**: system+tools+user -> function_call
- **Pair 2**: system+tools+user+observation -> function_call  
- **Pair 3**: system+tools+user+observation -> gpt

## 二、评估指标

### 2.1 基本统计指标
- **total_conversations**: 总对话数量 (397)
- **total_pairs**: 总评估对数量 (1191)

### 2.2 Pair级别指标

#### Pair 1 指标
- **pair1.total**: Pair 1 的总数量 (397)
- **pair1.accuracy**: Pair 1 的准确率 (0.358)
- **pair1.precision@1**: Pair 1 的工具名称精确匹配率 (0.589)

#### Pair 2 指标
- **pair2.total**: Pair 2 的总数量 (397)
- **pair2.accuracy**: Pair 2 的准确率 (0.699)
- **pair2.precision@1**: Pair 2 的工具名称精确匹配率 (0.814)
- **pair2_consider_recall.total**: 考虑recall条件的Pair 2数量 (129)
- **pair2_consider_recall.accuracy**: 考虑recall条件的Pair 2准确率 (0.754)
- **pair2_consider_recall.precision@1**: 考虑recall条件的Pair 2工具名称精确匹配率 (0.86)

#### Pair 3 指标
- **pair3.total**: Pair 3 的总数量 (397)
- **pair3.answer_score**: Pair 3 的文本生成质量得分 (0.51)

### 2.3 Recall指标
- **recall_metrics.total_pairs**: 参与recall评估的pair总数 (234)
- **recall_metrics.recall@5_1**: recall=1的pair数量 (129)
- **recall_metrics.recall@5_0**: recall=0的pair数量 (105)
- **recall_metrics.recall_rate**: recall率 (0.551)

### 2.4 总体指标
- **overall.total**: 总体评估对数量 (1191)
- **overall.accuracy**: 总体准确率 (0.529)
- **overall.precision@1**: 总体工具名称精确匹配率 (0.702)
- **overall.answer_score**: 总体文本生成质量得分 (0.51)

## 三、指标说明

### 3.1 准确率 (Accuracy)
- **计算原理**: 对于工具调用pair，比较预测的工具调用与目标工具调用的一致性，包括工具名称和参数匹配
- **评分规则**: 
  - 工具名称匹配: 50%权重
  - 参数匹配: 50%权重
  - 最终得分范围: 0-1

### 3.2 精确匹配率 (Precision@1)
- **计算原理**: 仅考虑工具名称是否完全匹配
- **评分规则**: 工具名称完全匹配得1分，否则得0分

### 3.3 文本生成质量得分 (Answer Score)
- **计算原理**: 使用Gemini-2.5-flash模型评估文本生成的四个维度
- **评估维度**:
  - 内容准确性 (0-10分)
  - 完整性 (0-10分)
  - 表达质量 (0-10分)
  - 格式一致性 (0-10分)
- **最终得分**: 四个维度的平均分除以10，范围0-1

### 3.4 Recall指标
- **计算原理**: 基于Pair 1的预测结果调用检索工具，检查Pair 2的目标工具是否在（调用retrieval_tool得到的）检索结果的前5个中
- **计算条件**: 仅在Pair 1的precision@1=1.0且有有效预测query时才计算
- **评分规则**: 目标工具在检索结果中得1分，否则得0分

### 3.5 考虑Recall的指标
- **pair2_consider_recall**: 仅在recall=1的条件下计算Pair 2的指标
- **意义**: 评估在工具检索成功情况下的模型表现

## 四、运行命令

python Data_evaluation.py.py


## 五、默认输入输出

### 5.1 输入文件
- **主要输入**: `data/9.17_evaluate_data_top5_final.json`
  - 包含待评估的对话数据，格式为conversations数组
  - 每个对话包含system、tools、conversations字段

#### 5.1.1 输入文件格式要求

**顶层结构**:
```json
[
  {
    "conversations": [...],
    "system": "...",
    "tools": "..." // 可选字段
  },
  ...
]
```

**必需字段**:
- **`conversations`** (数组，必需): 对话消息序列
  - 每个消息对象必须包含:
    - `"from"`: 字符串，必须是 `"human"`、`"function_call"`、`"observation"`、`"gpt"` 之一
    - `"value"`: 字符串，消息内容
- **`system`** (字符串，必需): 系统提示词，可包含 `<tools>` 标签

**可选字段**:
- **`tools`** (字符串，可选): JSON格式的工具定义，默认为 `"[]"`

#### 5.1.2 消息类型说明

| 消息类型 |       from值      |        用途       |                       示例                      |
|---------|-------------------|-------------------|------------------------------------------------|
| 用户输入 |      `"human"`    |  用户的查询或请求  | `{"from": "human", "value": "帮我查询收入数据"}` |
| 工具调用 | `"function_call"` | AI调用的工具和参数 | `{"from": "function_call", "value": "{\"name\":\"tool_name\",\"arguments\":{...}}"}` |
| 工具返回 |  `"observation"`  |   工具执行的结果   | `{"from": "observation", "value": "工具返回的结果数据"}` |
|  AI回复 |       `"gpt"`      |    AI的最终回答   | `{"from": "gpt", "value": "基于工具结果的回答"}` |

#### 5.1.3 对话序列要求

脚本会按照以下逻辑解析conversations来生成评估对：

1. **Pair 1**: `human` → `function_call`
   - 从 `"human"` 消息开始，寻找紧随其后的 `"function_call"` 消息
   
2. **Pair 2**: `human` + `observation` → `function_call`
   - 从 `"observation"` 消息开始，寻找紧随其后的 `"function_call"` 消息
   
3. **Pair 3**: `human` + `observation` → `gpt`
   - 从 `"observation"` 消息开始，寻找紧随其后的 `"gpt"` 消息

#### 5.1.4 格式验证

**严格程度**: 中等严格
- ✅ **必需字段**: `conversations`、`system` 必须存在且格式正确
- ✅ **字段类型**: 必须是正确的数据类型（数组、字符串）
- ✅ **枚举值**: `"from"` 字段必须是预定义的值
- ✅ **数组结构**: conversations必须是数组格式
- ⚠️ **可选字段**: `tools` 字段可选，有默认值
- ⚠️ **内容验证**: 不验证具体的工具调用内容格式

**常见错误**:
- `KeyError`: 缺少必需字段 `conversations` 或 `system`
- `TypeError`: 字段类型不正确（如conversations不是数组）
- `ValueError`: `"from"` 字段值不是预定义的枚举值

#### 5.1.5 完整示例

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "帮我看下2025年6月到8月各种收入类型的占比情况"
      },
      {
        "from": "function_call",
        "value": "{\"name\":\"retrieval_tool\",\"arguments\":{\"query\":\"收入分析\",\"source_filter\":\"toollist\",\"user_id\":136451106,\"top_k\":5}}"
      },
      {
        "from": "observation",
        "value": "[{\"name\":\"analyze_revenue_by_type\",\"description\":\"收入分析工具\"}]"
      },
      {
        "from": "function_call",
        "value": "{\"name\":\"analyze_revenue_by_type\",\"arguments\":{\"start_date\":\"2025-06-01\",\"end_date\":\"2025-08-31\"}}"
      },
      {
        "from": "observation",
        "value": "{\"result\":{\"收入\":\"1,285,893.00\",\"门票\":\"47.1%\",\"交通\":\"50.6%\"}}"
      },
      {
        "from": "gpt",
        "value": "根据2025年6月到8月的收入数据分析，各种收入类型的占比情况如下：交通收入占50.6%，门票收入占47.1%..."
      }
    ],
    "system": "# 工具\n\n你可以调用一个或多个函数来协助处理用户查询。\n\n在 <tools></tools> XML 标签中提供了可用的函数签名：\n<tools>\n</tools>",
    "tools": "[{\"name\":\"retrieval_tool\",\"description\":\"检索工具\"}]"
  }
]
```

**从上述示例生成的评估对**:
- **Pair 1**: `system + user查询` → `retrieval_tool调用`
- **Pair 2**: `system + user查询 + 检索结果` → `analyze_revenue_by_type调用`  
- **Pair 3**: `system + user查询 + 分析结果` → `最终回答`

### 5.2 输出文件
- **主要输出**: `metrics/data_evaluation_results.json`
  - 完整的评估报告，包含详细结果和指标统计
- **实时指标**: `metrics/realtime_metrics.json`
  - 实时更新的指标数据，每处理一个评估对就更新
- **日志文件**: `metrics/data_evaluation.log`
  - 详细的运行日志，包含评估进度和错误信息
- **断点文件**: `metrics/evaluation_checkpoint.json`
  - 支持断点续传的中间状态文件（完成后自动删除）

## 六、技术特性

### 6.1 断点续传
- 支持中断后从上次位置继续评估
- 自动保存处理进度到checkpoint文件
- 评估完成后自动清理断点文件

### 6.2 实时指标更新
- 每处理一个评估对就更新实时指标
- 可随时查看当前评估进度和指标状态

### 6.3 多模型支持
- **预测模型**: Qwen3-8B (通过API调用)
- **评估模型**: Gemini-2.5-flash (用于文本生成质量评估)
- **检索工具**: 调用retrieval_tool

### 6.4 容错机制
- API调用重试机制（最多3次）
- 指数退避策略
- 详细的错误日志记录

## 七、注意事项

1. **API配置**: 确保GEMINI_API_KEY和QWEN_API_URL配置正确
2. **网络连接**: 需要稳定的网络连接访问外部API服务
3. **存储空间**: 确保有足够空间存储输出文件和日志
4. **运行时间**: 完整评估可能需要数小时，建议后台运行
5. **数据格式**: 输入文件必须严格符合指定的JSON格式
   - 确保必需字段 `conversations` 和 `system` 存在
   - `conversations` 必须是数组，每个消息包含 `"from"` 和 `"value"` 字段
   - `"from"` 字段值必须是 `"human"`、`"function_call"`、`"observation"`、`"gpt"` 之一
   - 建议使用提供的示例文件作为模板
6. **对话序列**: 确保conversations中的消息序列符合评估逻辑要求
   - 每个 `"human"` 消息后应跟随 `"function_call"` 消息
   - 每个 `"observation"` 消息后应跟随 `"function_call"` 或 `"gpt"` 消息