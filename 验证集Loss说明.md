# SFT训练中验证集Loss说明

## 1. 验证集Loss是什么？

在SFT（Supervised Fine-Tuning）训练中：

- **训练集Loss**：模型在训练数据上计算的损失值，用于更新模型参数
- **验证集Loss（eval_loss）**：模型在验证数据集上计算的损失值，用于评估模型在未见过数据上的表现

验证集Loss的作用：
- 监控模型是否过拟合（训练loss下降但验证loss上升）
- 评估模型的泛化能力
- 帮助选择最佳的训练checkpoint

## 2. 如何启用和打印验证集Loss？

### 方法1：使用独立的验证数据集

在训练配置文件中（如`.yaml`文件）添加以下配置：

```yaml
### dataset
dataset: your_train_dataset
eval_dataset: your_eval_dataset  # 指定验证数据集

### eval
do_eval: true  # 启用评估（通常会自动启用）
eval_strategy: steps  # 评估策略：steps 或 epoch
eval_steps: 500  # 每500步评估一次（如果使用steps策略）
per_device_eval_batch_size: 1  # 验证时的batch size
```

### 方法2：从训练集中划分验证集

如果只有一个数据集，可以从训练集中划分一部分作为验证集：

```yaml
### dataset
dataset: your_dataset
val_size: 0.1  # 从训练集中划分10%作为验证集

### eval
eval_strategy: steps
eval_steps: 500
per_device_eval_batch_size: 1
```

**注意**：`eval_dataset`和`val_size`不能同时使用。

## 3. 验证集Loss的输出位置

### 3.1 训练日志

验证集Loss会在训练过程中自动打印到控制台，格式类似：

```
{'loss': 0.5, 'eval_loss': 0.6, 'epoch': 1.0}
```

### 3.2 Metrics文件

验证集Loss会保存在训练输出目录的`trainer_state.json`和`all_results.json`文件中：

- `trainer_state.json`：包含完整的训练历史，包括每个评估步骤的`eval_loss`
- `all_results.json`：包含最终的训练和评估指标

### 3.3 Loss曲线图

如果设置了`plot_loss: true`，验证集Loss会被绘制在loss曲线图中：

```yaml
plot_loss: true  # 会生成包含train loss和eval loss的曲线图
```

生成的图片保存在`output_dir/training_loss.png`中。

## 4. 代码实现位置

验证集Loss的计算和打印主要在以下文件中实现：

1. **`src/llamafactory/train/sft/workflow.py`** (第121-124行)
   ```python
   if training_args.do_eval:
       metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
       trainer.log_metrics("eval", metrics)
       trainer.save_metrics("eval", metrics)
   ```

2. **`src/llamafactory/train/callbacks.py`** (第280行)
   ```python
   eval_loss=state.log_history[-1].get("eval_loss"),
   ```

3. **`src/llamafactory/train/sft/workflow.py`** (第108-113行)
   ```python
   if isinstance(dataset_module.get("eval_dataset"), dict):
       keys += sum(
           [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
       )
   else:
       keys += ["eval_loss", "eval_accuracy"]
   ```

## 5. 完整配置示例

```yaml
### model
model_name_or_path: your_model_path
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora

### dataset
dataset: train_data
eval_dataset: eval_data  # 或者使用 val_size: 0.1
template: your_template
cutoff_len: 2048

### output
output_dir: saves/your_model/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true  # 生成loss曲线图

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0

### eval
do_eval: true  # 启用评估
eval_strategy: steps  # 或 "epoch"
eval_steps: 500  # 每500步评估一次
per_device_eval_batch_size: 1
```

## 6. 常见问题

### Q: 为什么看不到验证集Loss？

A: 检查以下几点：
1. 是否设置了`eval_dataset`或`val_size`
2. 是否设置了`eval_strategy`（如`steps`或`epoch`）
3. 如果使用`steps`策略，确保`eval_steps`已设置
4. 检查训练日志中是否有评估相关的输出

### Q: 验证集Loss什么时候计算？

A: 根据`eval_strategy`设置：
- `steps`：每`eval_steps`步计算一次
- `epoch`：每个epoch结束时计算一次

### Q: 如何查看历史验证集Loss？

A: 查看`output_dir/trainer_state.json`文件，其中包含所有评估步骤的`eval_loss`记录。

## 7. 总结

- 验证集Loss是评估模型泛化能力的重要指标
- 通过配置`eval_dataset`或`val_size`来设置验证集
- 通过`eval_strategy`和`eval_steps`控制评估频率
- 验证集Loss会自动打印到日志、保存到metrics文件，并可绘制成曲线图
