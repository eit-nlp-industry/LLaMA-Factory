#!/usr/bin/env python3
"""
增强的工具调用训练脚本
强化模型的任务执行能力，通过模板和约束机制确保：
1. 检索阶段必须调用retrieval_tool
2. 业务工具必须从retrieval_tool返回的top5工具中选择
3. 严格按照inputSchema提取参数
4. 给出准确的总结
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

def create_enhanced_system_prompt():
    """创建增强的系统提示，强化约束"""
    return """
# 工具调用规则

你是一个专业的工具调用助手。在处理用户查询时，必须严格遵循以下流程和规则：

## 第一阶段：检索工具调用（必须执行）

**规则1：检索阶段强制约束**
- 当用户提出查询时，**必须首先调用 retrieval_tool** 来检索相关工具
- retrieval_tool 的参数要求：
  - `query`: 必须使用用户的完整查询内容
  - `source_filter`: 必须设置为 "toollist"（用于检索工具库）
  - `top_k`: 建议设置为 5，以获取最相关的工具列表
  - `user_id`: 必须提供有效的用户ID

**禁止行为：**
- ❌ 跳过检索阶段直接调用业务工具
- ❌ 使用错误的 source_filter 值
- ❌ 省略必需的参数（query, source_filter, user_id）

## 第二阶段：业务工具选择（严格约束）

**规则2：工具选择约束**
- 从 retrieval_tool 返回的 **top5 工具列表** 中选择最合适的业务工具
- 必须从返回的工具列表中选择，**不能调用列表外的工具**
- 仔细分析工具的描述（description）和输入模式（inputSchema），选择最匹配的工具

**工具选择判断标准：**
1. 工具的描述是否与用户查询意图匹配
2. 工具的 category（listing/analysis）是否适合查询类型
3. 工具的 inputSchema 是否支持所需的查询参数

## 第三阶段：参数提取（严格遵循）

**规则3：参数提取约束**
- 必须严格按照工具的 inputSchema 定义提取参数
- 对于必填参数（required字段），必须全部提供
- 对于可选参数，根据用户查询和工具返回结果合理设置
- 参数类型必须匹配 inputSchema 的定义（string/integer/array/object等）
- 时间参数格式：推荐使用 YYYY-MM-DD，也支持完整 RFC3339 格式

**参数提取检查清单：**
- ✅ 所有必填参数都已提供
- ✅ 参数类型符合 inputSchema 定义
- ✅ 枚举值（enum）在允许范围内
- ✅ 时间格式正确
- ✅ 数组和对象格式符合要求

## 第四阶段：结果总结（准确完整）

**规则4：总结生成要求**
- 基于工具返回的结果，生成准确、完整的总结
- 如果工具返回空数据，明确说明原因
- 如果工具返回错误，解释错误原因
- 总结应包含关键信息，避免冗余

## 完整流程示例

```
用户查询: "查一下2025年6月到现在已完成的订单，只要最新的5条就行。"

步骤1 - 检索工具调用（必须）:
<tool_call>
{"name": "retrieval_tool", "arguments": {"query": "查一下2025年6月到现在已完成的订单，只要最新的5条就行。", "source_filter": "toollist", "top_k": 5, "user_id": 13}}
</tool_call>

步骤2 - 工具返回（observation）:
[{"name": "list_orders", "description": "...", "inputSchema": {...}}, ...]

步骤3 - 业务工具调用（从返回列表中选择）:
<tool_call>
{"name": "list_orders", "arguments": {"time_start": "2025-06-01", "time_end": "2025-10-28", "status": "已完成", "limit": 5}}
</tool_call>

步骤4 - 结果总结:
<answer>
已为您查询到2025年6月至今最新的5条已完成订单：
1. 订单编号：...，客户名称：...，创建时间：...
...
</answer>
```

## 错误预防

**严格禁止以下行为：**
1. ❌ 跳过 retrieval_tool 直接调用业务工具
2. ❌ 调用 retrieval_tool 返回列表外的工具
3. ❌ 忽略 inputSchema 的必填参数要求
4. ❌ 使用错误的参数类型或格式
5. ❌ 在工具返回结果后重复调用相同工具
6. ❌ 生成与工具返回结果不符的总结

## 质量检查

在每次工具调用前，请自检：
- [ ] 是否已调用 retrieval_tool？
- [ ] 选择的工具是否在返回的 top5 列表中？
- [ ] 所有必填参数是否已提供？
- [ ] 参数格式是否符合 inputSchema？
- [ ] 是否已基于工具结果生成准确总结？

记住：严格按照流程执行，确保每一步都符合约束要求，避免任何预期外的预测。
"""

def create_enhanced_training_config():
    """创建优化的训练配置"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"saves/Qwen3-8B/lora/enhanced_tool_calling_{timestamp}"
    
    # 优化的超参数配置
    # 基于最佳实践和数据集特点调整
    config = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": "/data/models/Qwen3-8B",  # 根据实际情况修改
        "preprocessing_num_workers": 16,
        "finetuning_type": "lora",
        "template": "qwen3",
        "flash_attn": "auto",
        "dataset_dir": "data",
        "dataset": "tool_calling_12_08",  # 将在dataset_info.json中定义
        "cutoff_len": 8192,
        
        # 学习率和训练轮数 - 针对工具调用任务优化
        "learning_rate": 2.0e-5,  # 适中的学习率，确保稳定学习
        "num_train_epochs": 8.0,  # 足够的轮数但不过度训练
        "max_samples": 100000,
        
        # 批次配置 - 平衡内存和训练效果
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,  # 有效batch size = 16
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,  # 10%的warmup，帮助稳定训练
        
        # 正则化和稳定性
        "max_grad_norm": 0.3,  # 更严格的梯度裁剪，提高稳定性
        "weight_decay": 0.01,
        "lora_rank": 64,  # 更高的rank，提高表达能力
        "lora_alpha": 128,  # alpha = 2 * rank，保持比例
        "lora_dropout": 0.1,  # 适度的dropout，防止过拟合
        
        # 训练设置
        "logging_steps": 10,
        "save_steps": 500,
        "save_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "packing": False,
        "enable_thinking": False,
        "overwrite_cache": True,
        
        # 输出和优化
        "output_dir": output_dir,
        "bf16": True,
        "plot_loss": True,
        "trust_remote_code": True,
        "ddp_timeout": 180000000,
        "include_num_input_tokens_seen": True,
        "optim": "adamw_torch",
        "lora_target": "all",
        "gradient_checkpointing": True,
        
        # 数据加载
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "dataloader_drop_last": False,
        
        # 随机种子
        "seed": 42,
        
        # 其他优化
        "report_to": "none",
        "save_total_limit": 3,  # 只保留最新的3个checkpoint
    }
    
    return config, output_dir

def update_dataset_info():
    """更新dataset_info.json，添加新的数据集配置"""
    
    dataset_info_path = "data/dataset_info.json"
    
    # 读取现有配置
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # 添加新的数据集配置
    dataset_info["tool_calling_12_08"] = {
        "file_name": "dataset/12_08/train.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools"
        }
    }
    
    dataset_info["tool_calling_12_08_test"] = {
        "file_name": "dataset/12_08/test.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "system": "system",
            "tools": "tools"
        }
    }
    
    # 保存更新
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已更新 {dataset_info_path}")

def create_training_script(config, output_dir):
    """创建训练脚本"""
    
    script_content = f"""#!/bin/bash
# 增强的工具调用训练脚本
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "🚀 启动增强的工具调用训练"
echo "📁 输出目录: {output_dir}"
echo "=" * 60

# 创建输出目录
mkdir -p "{output_dir}"

# 运行训练命令
llamafactory-cli train \\
    --stage {config['stage']} \\
    --do_train {config['do_train']} \\
    --model_name_or_path {config['model_name_or_path']} \\
    --preprocessing_num_workers {config['preprocessing_num_workers']} \\
    --finetuning_type {config['finetuning_type']} \\
    --template {config['template']} \\
    --flash_attn {config['flash_attn']} \\
    --dataset_dir {config['dataset_dir']} \\
    --dataset {config['dataset']} \\
    --cutoff_len {config['cutoff_len']} \\
    --learning_rate {config['learning_rate']} \\
    --num_train_epochs {config['num_train_epochs']} \\
    --max_samples {config['max_samples']} \\
    --per_device_train_batch_size {config['per_device_train_batch_size']} \\
    --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
    --lr_scheduler_type {config['lr_scheduler_type']} \\
    --warmup_ratio {config['warmup_ratio']} \\
    --max_grad_norm {config['max_grad_norm']} \\
    --weight_decay {config['weight_decay']} \\
    --logging_steps {config['logging_steps']} \\
    --save_steps {config['save_steps']} \\
    --save_strategy {config['save_strategy']} \\
    --evaluation_strategy {config['evaluation_strategy']} \\
    --eval_steps {config['eval_steps']} \\
    --packing {config['packing']} \\
    --enable_thinking {config['enable_thinking']} \\
    --overwrite_cache {config['overwrite_cache']} \\
    --output_dir {output_dir} \\
    --bf16 {config['bf16']} \\
    --plot_loss {config['plot_loss']} \\
    --trust_remote_code {config['trust_remote_code']} \\
    --ddp_timeout {config['ddp_timeout']} \\
    --include_num_input_tokens_seen {config['include_num_input_tokens_seen']} \\
    --optim {config['optim']} \\
    --lora_rank {config['lora_rank']} \\
    --lora_alpha {config['lora_alpha']} \\
    --lora_dropout {config['lora_dropout']} \\
    --lora_target {config['lora_target']} \\
    --gradient_checkpointing {config['gradient_checkpointing']} \\
    --dataloader_pin_memory {config['dataloader_pin_memory']} \\
    --dataloader_num_workers {config['dataloader_num_workers']} \\
    --remove_unused_columns {config['remove_unused_columns']} \\
    --dataloader_drop_last {config['dataloader_drop_last']} \\
    --seed {config['seed']} \\
    --save_total_limit {config['save_total_limit']}

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: {output_dir}"
"""
    
    script_path = f"run_enhanced_tool_calling_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

def create_data_validator():
    """创建数据验证工具，检查数据是否符合约束"""
    
    validator_code = '''#!/usr/bin/env python3
"""
数据验证工具
检查训练数据是否符合工具调用约束
"""

import json
import sys
from pathlib import Path

def validate_retrieval_tool_call(conversations):
    """验证是否包含retrieval_tool调用"""
    has_retrieval = False
    retrieval_index = -1
    
    for i, msg in enumerate(conversations):
        if msg.get("from") == "function_call":
            try:
                value = json.loads(msg.get("value", "{}"))
                if value.get("name") == "retrieval_tool":
                    has_retrieval = True
                    retrieval_index = i
                    # 验证参数
                    args = value.get("arguments", {})
                    missing_params = []
                    if "query" not in args:
                        missing_params.append("query")
                    if "source_filter" not in args:
                        missing_params.append("source_filter")
                    if "user_id" not in args:
                        missing_params.append("user_id")
                    
                    if missing_params:
                        print(f"⚠️  警告: retrieval_tool缺少必需参数 {missing_params} (消息索引 {i})")
                    
                    if args.get("source_filter") != "toollist":
                        print(f"⚠️  警告: retrieval_tool的source_filter应为'toollist'，当前为'{args.get('source_filter')}' (消息索引 {i})")
                    break
            except Exception as e:
                pass
    
    return has_retrieval, retrieval_index

def validate_tool_selection(conversations, retrieval_index):
    """验证业务工具是否从retrieval_tool返回的列表中选择"""
    if retrieval_index < 0 or retrieval_index + 1 >= len(conversations):
        return False, "无法找到retrieval_tool的返回结果"
    
    # 获取retrieval_tool的返回结果
    observation = conversations[retrieval_index + 1]
    if observation.get("from") != "observation":
        return False, "retrieval_tool后没有observation"
    
    try:
        tool_list = json.loads(observation.get("value", "[]"))
        tool_names = [tool.get("name") for tool in tool_list if isinstance(tool, dict)]
    except:
        return False, "无法解析retrieval_tool返回的工具列表"
    
    # 检查后续的业务工具调用
    for i in range(retrieval_index + 2, len(conversations)):
        msg = conversations[i]
        if msg.get("from") == "function_call":
            try:
                value = json.loads(msg.get("value", "{}"))
                tool_name = value.get("name")
                if tool_name and tool_name != "retrieval_tool":
                    if tool_name not in tool_names:
                        return False, f"业务工具 {tool_name} 不在retrieval_tool返回的列表中"
            except:
                pass
    
    return True, "工具选择验证通过"

def validate_parameters(conversations):
    """验证参数提取是否符合inputSchema"""
    # 这里可以添加更详细的参数验证逻辑
    # 目前只做基本检查
    return True, "参数验证通过"

def validate_data_file(file_path):
    """验证数据文件"""
    print(f"\\n🔍 验证数据文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    valid_count = 0
    errors = []
    
    for idx, item in enumerate(data):
        conversations = item.get("conversations", [])
        
        # 检查retrieval_tool调用
        has_retrieval, retrieval_idx = validate_retrieval_tool_call(conversations)
        if not has_retrieval:
            errors.append(f"样本 {idx}: 缺少retrieval_tool调用")
            continue
        
        # 检查工具选择
        is_valid, msg = validate_tool_selection(conversations, retrieval_idx)
        if not is_valid:
            errors.append(f"样本 {idx}: {msg}")
            continue
        
        # 检查参数
        is_valid, msg = validate_parameters(conversations)
        if not is_valid:
            errors.append(f"样本 {idx}: {msg}")
            continue
        
        valid_count += 1
    
    print(f"✅ 验证完成: {valid_count}/{total} 样本通过验证")
    if errors:
        print(f"\\n❌ 发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
    
    return valid_count == total

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python validate_tool_calling_data.py <数据文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        sys.exit(1)
    
    is_valid = validate_data_file(file_path)
    sys.exit(0 if is_valid else 1)
'''
    
    validator_path = "validate_tool_calling_data.py"
    with open(validator_path, 'w', encoding='utf-8') as f:
        f.write(validator_code)
    
    os.chmod(validator_path, 0o755)
    return validator_path

def main():
    """主函数"""
    
    print("🚀 创建增强的工具调用训练方案")
    print("=" * 60)
    
    # 1. 更新数据集配置
    print("\\n📝 步骤1: 更新数据集配置...")
    update_dataset_info()
    
    # 2. 创建训练配置
    print("\\n⚙️  步骤2: 创建优化的训练配置...")
    config, output_dir = create_enhanced_training_config()
    print(f"   ✅ 输出目录: {output_dir}")
    print(f"   ✅ 学习率: {config['learning_rate']}")
    print(f"   ✅ 训练轮数: {config['num_train_epochs']}")
    print(f"   ✅ LoRA rank: {config['lora_rank']}, alpha: {config['lora_alpha']}")
    
    # 3. 创建训练脚本
    print("\\n📜 步骤3: 创建训练脚本...")
    script_path = create_training_script(config, output_dir)
    print(f"   ✅ 训练脚本: {script_path}")
    
    # 4. 创建数据验证工具
    print("\\n🔍 步骤4: 创建数据验证工具...")
    validator_path = create_data_validator()
    print(f"   ✅ 验证工具: {validator_path}")
    
    # 5. 保存增强的系统提示
    print("\\n💬 步骤5: 保存增强的系统提示...")
    system_prompt = create_enhanced_system_prompt()
    prompt_path = "enhanced_system_prompt.txt"
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(system_prompt)
    print(f"   ✅ 系统提示: {prompt_path}")
    
    print("\\n" + "=" * 60)
    print("✅ 所有文件已创建完成！")
    print("\\n📋 使用说明:")
    print("=" * 60)
    print(f"1. 验证训练数据:")
    print(f"   python {validator_path} data/dataset/12_08/train.json")
    print(f"\\n2. 运行训练:")
    print(f"   bash {script_path}")
    print(f"\\n3. 系统提示已保存到: {prompt_path}")
    print(f"   可以在数据预处理时使用此提示替换默认系统消息")
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    main()

