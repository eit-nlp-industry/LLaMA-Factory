# Token调试日志系统使用指南

## 🎯 功能概述

这个调试系统为LLaMA-Factory的ShareGPT格式训练添加了详细的token长度跟踪，帮助开发者理解：
- 数据样本的token分布
- 截断策略的执行过程  
- cutoff_len的使用效率

## 📁 文件结构

```
LLaMA-Factory/
├── configure_token_logs.py          # 日志配置脚本
├── token_debug_YYYYMMDD_HHMMSS.log  # 统一调试日志文件（自动生成）
├── src/llamafactory/data/
│   ├── processor/
│   │   ├── supervised.py            # 数据样本处理日志
│   │   └── processor_utils.py       # 截断策略日志
│   └── template.py                  # 模板编码日志
└── data/dataset/
    └── function_call_training_explanation.md  # 详细文档
```

## 🚀 快速开始

### 1. 运行训练（自动生成日志）

```bash
# 正常训练，日志会自动保存到 token_debug_*.log
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    --stage sft \
    --model_name_or_path /path/to/model \
    --dataset your_dataset \
    --cutoff_len 8192 \
    [其他参数...]
```

### 2. 查看调试日志

```bash
# 实时查看所有调试信息
tail -f token_debug_*.log

# 查看数据样本处理信息
grep "TOKEN_DEBUG" token_debug_*.log

# 查看截断策略执行
grep "INFER_SEQLEN" token_debug_*.log

# 查看模板编码信息
grep "TEMPLATE_DEBUG" token_debug_*.log
```

## 📊 日志类型说明

### [TOKEN_DEBUG] - 数据样本处理
- 原始conversations长度
- 编码后的pairs数量
- 每个pair的token长度
- 最终使用率和截断情况

### [TEMPLATE_DEBUG] - 模板编码
- 输入messages数量
- 编码后messages数量
- 生成的pairs数量
- 每个pair的source/target长度

### [INFER_SEQLEN] - 截断策略
- 输入参数（source_len, target_len, cutoff_len）
- 选择的截断策略（策略1/2/3）
- 截断后的长度

## 🔍 关键指标解读

### 使用率 (Usage Rate)
```
使用率 = 最终total_length / cutoff_len
```
- **>90%**: 高效利用，数据长度适中
- **70-90%**: 正常使用，可能有轻微浪费
- **<70%**: 使用率低，考虑增加cutoff_len或数据长度

### 截断事件
- **策略1**: target完全保留，截断source
- **策略2**: source完全保留，截断target  
- **策略3**: 同时截断source和target

## 📈 优化建议

### 如果使用率过低 (<70%)
```bash
# 增加cutoff_len
--cutoff_len 12288

# 或增加数据长度
# 预处理时保留更多observation内容
```

### 如果经常截断
```bash
# 减少cutoff_len提高效率
--cutoff_len 6144

# 或预处理压缩数据
# 减少function_call结果的长度
```

## 🛠️ 高级用法

### 分析特定样本
```bash
# 查看截断的样本
grep -A5 -B5 "截断" token_debug_*.log

# 统计截断频率
grep "截断" token_debug_*.log | wc -l
```

### 性能监控
```bash
# 监控使用率趋势
grep "使用率" token_debug_*.log | tail -20

# 查看最长的样本
grep "total_length" token_debug_*.log | sort -k3 -nr | head -10
```

## ⚠️ 注意事项

1. **性能影响**: 调试日志会轻微增加训练时间
2. **存储空间**: 日志文件可能较大，建议定期清理
3. **生产环境**: 生产训练时建议关闭调试日志
4. **日志轮转**: 文件达到50MB会自动轮转，保留3天历史

## 🔧 故障排除

### 日志文件未生成
```bash
# 检查loguru是否安装
pip install loguru

# 手动测试日志配置
python configure_token_logs.py
```

### 日志内容为空
- 确保训练数据包含function_call格式
- 检查cutoff_len设置是否合理
- 确认模板配置正确

## 📞 支持

如有问题，请查看：
- 详细文档：`data/dataset/function_call_training_explanation.md`
- 日志文件：`token_debug_*.log`
- 配置脚本：`configure_token_logs.py`
