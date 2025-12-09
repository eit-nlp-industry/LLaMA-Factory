# CUDA OOM 问题分析与解决方案

## 问题诊断

### 当前状态
- GPU 总容量: 39.49 GiB
- 已使用: 36.94 GiB (当前进程)
- 其他进程占用: 2.34 GiB
- **内存碎片化**: 19.52 GiB 被保留但未分配
- 尝试分配: 18.55 GiB (失败)

### 关键配置
- 模型: Qwen3-8B
- 序列长度: 8192 (非常长)
- Batch size: 1 (已最小)
- 梯度累积: 8
- 梯度检查点: 已启用

## 解决方案（按优先级）

### 方案 1: 解决内存碎片化（推荐优先尝试）
**问题**: PyTorch 内存分配器导致严重碎片化（19.52 GiB 未使用）

**解决方法**: 设置环境变量
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**优点**: 
- 不需要修改训练配置
- 可能直接解决问题
- 不影响训练效果

**操作**: 在运行训练前设置此环境变量，或在脚本中设置

---

### 方案 2: 清理其他进程占用
**问题**: 进程 2737260 占用 2.34 GiB

**解决方法**: 
```bash
# 查看占用 GPU 的进程
nvidia-smi

# 如果进程 2737260 不是必需的，可以终止
kill 2737260
```

**优点**: 释放额外内存
**注意**: 确保该进程不是其他重要任务

---

### 方案 3: 减少序列长度（如果数据允许）
**问题**: `cutoff_len=8192` 对 8B 模型来说很长，占用大量内存

**内存占用估算**:
- 8192 序列长度: ~18-20 GiB
- 4096 序列长度: ~9-10 GiB (减少约 50%)
- 2048 序列长度: ~4-5 GiB (减少约 75%)

**解决方法**: 如果您的数据不需要这么长的序列，可以降低 `CUTOFF_LEN`

**检查方法**: 查看数据集中实际序列长度分布
```python
# 可以运行这个脚本检查数据长度
import json
with open('data/dataset/12_08/train.json', 'r') as f:
    data = json.load(f)
    lengths = [len(item['instruction'] + item.get('input', '') + item.get('output', '')) for item in data]
    print(f"最大长度: {max(lengths)}")
    print(f"平均长度: {sum(lengths)/len(lengths):.0f}")
    print(f"95%分位数: {sorted(lengths)[int(len(lengths)*0.95)]}")
```

---

### 方案 4: 使用 DeepSpeed ZeRO（如果支持）
**问题**: 需要进一步减少内存占用

**解决方法**: 使用 DeepSpeed ZeRO-2 或 ZeRO-3 进行内存优化

**注意**: 需要配置 DeepSpeed，可能影响训练速度

---

### 方案 5: 使用 QLoRA（量化 LoRA）
**问题**: 基础模型占用大量内存

**解决方法**: 使用 4-bit 或 8-bit 量化加载模型

**优点**: 大幅减少基础模型内存占用
**缺点**: 可能略微影响模型精度

---

## 推荐执行顺序

1. **首先尝试方案 1** (设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
   - 最简单，可能直接解决问题
   
2. **如果方案 1 不行，尝试方案 2** (清理其他进程)
   - 释放额外内存
   
3. **如果还不行，检查数据长度后考虑方案 3** (降低序列长度)
   - 如果数据确实不需要 8192，这是最有效的方案
   
4. **最后考虑方案 4 或 5** (DeepSpeed 或 QLoRA)
   - 需要更多配置，但可以处理更大的模型/序列

## 快速修复（最小改动）

如果只想快速尝试，可以在训练脚本中添加：

```python
# 在设置 CUDA_VISIBLE_DEVICES 之后添加
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

这不会改变任何训练配置，只是优化内存分配策略。


