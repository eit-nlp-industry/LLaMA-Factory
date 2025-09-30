# 异步并发评测脚本使用指南

## 概述

评测脚本已经升级为异步并发版本，大幅提升了评测速度。主要改进包括：

1. **异步API调用**：使用 `aiohttp` 替代 `requests` 进行异步HTTP请求
2. **并发处理**：支持多个对话和pairs的并发评估
3. **可配置并发数**：通过命令行参数控制并发级别
4. **保持兼容性**：保持原有的断点续传和实时指标功能

## 性能提升

- **API调用并发**：多个API请求可以同时进行
- **对话并发**：多个对话可以并行处理
- **Pair并发**：同一对话内的多个pairs可以并发评估
- **预期速度提升**：3-10倍（取决于API响应时间和并发配置）

## 新增命令行参数

```bash
--max_concurrent_conversations INT    最大并发对话数（默认: 5）
--max_concurrent_pairs INT           最大并发pair数（默认: 10）
--max_concurrent_api_calls INT       最大并发API调用数（默认: 20）
```

## 使用示例

### 基本使用（使用默认并发配置）
```bash
python data_evaluation.py \
    --input_file data/dataset/demo.json \
    --output_file evaluation/result.json \
    --start_idx 0 \
    --end_idx 100
```

### 高并发配置（适合性能较强的服务器）
```bash
python data_evaluation.py \
    --input_file data/dataset/demo.json \
    --output_file evaluation/result.json \
    --max_concurrent_conversations 10 \
    --max_concurrent_pairs 20 \
    --max_concurrent_api_calls 50 \
    --start_idx 0 \
    --end_idx 100
```

### 保守并发配置（适合资源受限的环境）
```bash
python data_evaluation.py \
    --input_file data/dataset/demo.json \
    --output_file evaluation/result.json \
    --max_concurrent_conversations 2 \
    --max_concurrent_pairs 5 \
    --max_concurrent_api_calls 10 \
    --start_idx 0 \
    --end_idx 100
```

### 多模型评估
```bash
python data_evaluation.py \
    --input_file data/dataset/demo.json \
    --models "my_lora,/data/models/Qwen3-8B" \
    --max_concurrent_conversations 3 \
    --max_concurrent_pairs 8 \
    --max_concurrent_api_calls 15
```

## 并发配置建议

### 根据服务器配置调整

**高性能服务器（16+ CPU核心，32+ GB内存）**
```bash
--max_concurrent_conversations 10-15
--max_concurrent_pairs 20-30
--max_concurrent_api_calls 50-100
```

**中等性能服务器（8 CPU核心，16 GB内存）**
```bash
--max_concurrent_conversations 5-8
--max_concurrent_pairs 10-15
--max_concurrent_api_calls 20-30
```

**低性能服务器（4 CPU核心，8 GB内存）**
```bash
--max_concurrent_conversations 2-3
--max_concurrent_pairs 5-8
--max_concurrent_api_calls 10-15
```

### 根据API服务能力调整

如果API服务有并发限制，建议：
- 降低 `max_concurrent_api_calls`
- 相应调整 `max_concurrent_conversations` 和 `max_concurrent_pairs`

## 环境变量支持

也可以通过环境变量设置并发配置：

```bash
export MAX_CONCURRENT_CONVERSATIONS=8
export MAX_CONCURRENT_PAIRS=15
export MAX_CONCURRENT_API_CALLS=30

python data_evaluation.py --input_file data/dataset/demo.json
```

## 注意事项

1. **内存使用**：并发数越高，内存使用越多
2. **API限制**：确保API服务能处理配置的并发请求数
3. **网络稳定性**：高并发可能对网络稳定性要求更高
4. **错误处理**：异步版本有完善的错误处理和重试机制
5. **断点续传**：异步版本完全支持断点续传功能

## 监控和调试

异步版本提供详细的日志输出，包括：
- 并发配置信息
- 每个对话和pair的处理状态
- API调用统计
- 错误和重试信息

可以通过日志监控评估进度和性能表现。


