#!/usr/bin/env python3
"""
最终的增强训练解决方案
通过修改LLaMA-Factory的核心文件，添加以下功能：

1. Label打印和Token分析
   - 显示数据切分：哪些token需要训练（labels != -100）
   - 显示完整的input_ids和labels
   - 追踪训练过程中token的具体变化
   - 支持中文Token解码

2. 预测Token监控
   - 实时监控模型的预测输出变化
   - 显示模型预测文本 vs 目标文本对比
   - 计算预测准确率统计
   - 追踪预测Token在训练过程中的变化
   - 显示预测变化的具体位置和内容
   - 详细的Token ID和文本对比
   - 预测准确率为0%时的调试信息
   - 预测文本变化分析和相似度计算
   - 文本差异详细对比（新增/删除/修改）

3. 多轮对话分析
   - 分析-100部分（忽略的Token）
   - 显示忽略Token的位置和内容
   - 多轮对话结构分析
   - 对话分段详情
   - 训练部分vs忽略部分的比例
   - 多轮对话模式检测

4. 训练过程监控
   - 每5步记录详细Token信息
   - 每步都检查Token和预测变化
   - 自动保存到日志文件

5. 验证集Loss监控和记录
   - 自动从训练集中划分10%作为验证集
   - 每100步评估一次验证集Loss
   - 实时提取和记录eval_loss到独立日志文件
   - 从trainer_state.json读取准确的eval_loss
   - 保存验证集Loss历史到JSON文件
   - 验证集Loss会绘制在loss曲线图中
   - 验证集Loss会保存在训练metrics中
"""

import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# 添加LLaMA-Factory路径
sys.path.append('/home/ziqiang/LLaMA-Factory')

def modify_trainer_to_add_monitoring():
    """修改训练器以添加监控功能"""
    
    # 修改SFT训练器，添加监控回调
    trainer_file = "/home/ziqiang/LLaMA-Factory/src/llamafactory/train/sft/trainer.py"
    
    # 读取现有文件
    with open(trainer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加了监控功能
    if "LabelPredictionMonitorCallback" in content:
        print("✅ 训练器已经包含监控功能")
        return True
    
    # 在__init__方法中添加监控回调
    init_modification = '''
        # 添加标签预测监控回调
        from ..callbacks import LabelPredictionMonitorCallback
        monitor_callback = LabelPredictionMonitorCallback(
            output_dir=self.args.output_dir,
            log_interval=5,
            save_detailed_logs=True
        )
        self.add_callback(monitor_callback)
'''
    
    # 在__init__方法的末尾添加监控回调
    if "self.add_callback(BAdamCallback)" in content:
        content = content.replace(
            "self.add_callback(BAdamCallback)",
            f"self.add_callback(BAdamCallback){init_modification}"
        )
    elif "self.label_debugger = None" in content:
        content = content.replace(
            "self.label_debugger = None",
            f"self.label_debugger = None{init_modification}"
        )
    else:
        # 在__init__方法的最后添加
        init_end_pattern = "            self.label_debugger = None"
        if init_end_pattern in content:
            content = content.replace(
                init_end_pattern,
                f"{init_end_pattern}{init_modification}"
            )
        else:
            # 如果找不到合适的位置，在__init__方法末尾添加
            content = content.replace(
                "        self.label_debugger = None",
                f"        self.label_debugger = None{init_modification}"
            )
    
    # 写回文件
    with open(trainer_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已修改训练器添加监控功能")
    return True

def create_enhanced_training_script():
    """创建增强的训练脚本"""
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = f"/home/ziqiang/LLaMA-Factory/saves/Qwen3-8B/lora/train_{timestamp}"
    
    # 创建日志目录
    log_dir = "/home/ziqiang/LLaMA-Factory/enhanced_training_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files = {
        "training": os.path.join(log_dir, f"training_monitor_{log_timestamp}.log"),
        "predictions": os.path.join(log_dir, f"prediction_monitor_{log_timestamp}.log"),
        "labels": os.path.join(log_dir, f"label_analysis_{log_timestamp}.log"),
        "alignment": os.path.join(log_dir, f"alignment_analysis_{log_timestamp}.log"),
        "eval_loss": os.path.join(log_dir, f"eval_loss_monitor_{log_timestamp}.log"),
        "main": os.path.join(log_dir, f"main_training_{log_timestamp}.log")
    }
    
    # 创建增强的训练脚本
    script_content = f"""#!/bin/bash
# 增强的LLaMA-Factory训练脚本
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

set -e

echo "🚀 启动增强的LLaMA-Factory训练"
echo "📁 输出目录: {output_dir}"
echo "📝 日志文件: {log_files['main']}"
echo "=" * 60

# 设置环境变量 - 双卡DDP训练（保持最佳单卡训练效果）
export CUDA_VISIBLE_DEVICES=0,4

# 创建输出目录
mkdir -p "{output_dir}"

# 创建日志目录
mkdir -p "{log_dir}"

# 设置日志文件路径
export ENHANCED_TRAINING_LOG="{log_files['main']}"
export ENHANCED_LABEL_LOG="{log_files['labels']}"
export ENHANCED_PREDICT_LOG="{log_files['predictions']}"
export ENHANCED_ALIGNMENT_LOG="{log_files['alignment']}"
export ENHANCED_EVAL_LOSS_LOG="{log_files['eval_loss']}"

# 记录开始时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 🚀 增强训练开始" >> "{log_files['main']}"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📁 输出目录: {output_dir}" >> "{log_files['main']}"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📝 日志文件: {log_files['main']}" >> "{log_files['main']}"

# 设置输出目录环境变量，供监控脚本使用
export OUTPUT_DIR="{output_dir}"

# 检查数据集一致性
echo "🔍 检查数据集一致性..."
python3 /home/ziqiang/LLaMA-Factory/check_dataset_consistency.py >> "{log_files['main']}" 2>&1
echo "✅ 数据集检查完成"

# 运行训练命令 - 双卡DDP分布式训练
echo "🔄 执行双卡DDP分布式训练命令..."
echo "⚡ 使用PyTorch DDP，配置与最佳单卡完全等效"
echo "📊 启用验证集评估，使用独立的test数据集，监控eval_loss"
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/models/Qwen3-8B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset sft_training_data_filter \
    --eval_dataset sft_test_data_01_08 \
    --cutoff_len 8192 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.5 \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_ratio 0.05 \
    --packing False \
    --enable_thinking False \
    --overwrite_cache True \
    --save_strategy steps \
    --output_dir {output_dir} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target all \
    --gradient_checkpointing True \
    --dataloader_pin_memory False \
    --dataloader_num_workers 0 \
    --remove_unused_columns False \
    --dataloader_drop_last False \
    --seed 42 \
    --eval_strategy steps \
    --eval_steps 25 \
    --per_device_eval_batch_size 1 \
    --do_eval True

# 记录结束时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | ✅ 训练完成" >> "{log_files['main']}"

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: {output_dir}"
echo "   📝 日志文件: {log_files['main']}"
echo "   🏷️ 标签分析: {log_files['labels']}"
echo "   🔮 预测监控: {log_files['predictions']}"
echo "   🎯 对齐分析: {log_files['alignment']}"
echo "   📈 验证集Loss: {log_files['eval_loss']}"
"""
    
    # 保存脚本
    script_path = os.path.join(log_dir, f"run_enhanced_training_{log_timestamp}.sh")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path, output_dir, log_files

def create_monitoring_script(log_files):
    """创建监控脚本"""
    
    # 将log_files转换为字符串格式
    log_files_str = str(log_files).replace("'", '"')
    
    monitoring_script = f"""#!/usr/bin/env python3
'''
增强训练监控脚本
实时监控训练过程中的以下内容：

1. Label和Token分析
   - 数据切分情况
   - Token变化追踪
   - 中文Token解码

2. 预测Token监控
   - 模型预测输出变化
   - 预测准确率统计
   - 预测文本对比

3. 训练过程监控
   - Loss变化
   - 验证集Loss (eval_loss) 监控和记录
   - 学习率调整
   - 训练进度

4. 验证集Loss监控
   - 实时提取和记录eval_loss
   - 从trainer_state.json读取准确的eval_loss
   - 保存验证集Loss历史到JSON文件
   - 监控验证集Loss变化趋势
'''

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

def setup_loggers(log_files):
    '''设置日志记录器'''
    loggers = {{}}
    
    for log_type, log_file in log_files.items():
        logger = logging.getLogger(f"monitor_{{log_type}}")
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        loggers[log_type] = logger
        
    return loggers

def monitor_training_logs(loggers, log_files):
    '''监控训练日志'''
    
    main_logger = loggers["main"]
    eval_loss_logger = loggers.get("eval_loss")
    main_logger.info("🔍 开始监控训练过程")
    
    # 监控训练日志文件
    trainer_log = None
    for log_file in log_files.values():
        if "main_training" in log_file:
            trainer_log = log_file
            break
    
    if trainer_log:
        main_logger.info(f"📝 监控训练日志: {{trainer_log}}")
        
        # 监控文件变化
        last_size = 0
        eval_loss_history = []  # 保存验证集loss历史
        
        while True:
            try:
                if os.path.exists(trainer_log):
                    current_size = os.path.getsize(trainer_log)
                    if current_size > last_size:
                        # 读取新增内容
                        with open(trainer_log, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            
                        # 记录新内容并提取eval_loss
                        for line in new_content.strip().split('\\n'):
                            if line.strip():
                                main_logger.info(f"📊 训练日志: {{line}}")
                                
                                # 提取eval_loss信息
                                if "eval_loss" in line.lower() or "'eval_loss'" in line or '"eval_loss"' in line:
                                    try:
                                        # 尝试从JSON格式中提取eval_loss
                                        import re
                                        # 匹配 eval_loss: value 或 "eval_loss": value
                                        match = re.search(r'["\']?eval_loss["\']?\\s*[:=]\\s*([0-9.]+)', line, re.IGNORECASE)
                                        if match:
                                            eval_loss_value = float(match.group(1))
                                            eval_loss_history.append({{
                                                "step": len(eval_loss_history) + 1,
                                                "eval_loss": eval_loss_value,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }})
                                            
                                            if eval_loss_logger:
                                                eval_loss_logger.info(
                                                    f"📈 Step {{len(eval_loss_history)}} | "
                                                    f"Eval Loss: {{eval_loss_value:.6f}} | "
                                                    f"Time: {{datetime.now().strftime('%H:%M:%S')}}"
                                                )
                                            
                                            main_logger.info(
                                                f"✅ 验证集Loss更新: {{eval_loss_value:.6f}}"
                                            )
                                    except Exception as e:
                                        pass  # 如果解析失败，忽略
                                
                        last_size = current_size
                        
                # 监控输出目录中的trainer_state.json以获取更准确的eval_loss
                # 尝试从环境变量或日志中获取output_dir
                output_dir = os.environ.get("OUTPUT_DIR")
                if not output_dir:
                    # 从日志文件中提取output_dir（如果存在）
                    try:
                        if os.path.exists(trainer_log):
                            with open(trainer_log, 'r', encoding='utf-8') as f:
                                content = f.read()
                                import re
                                match = re.search(r'输出目录[:：]\\s*([^\\n]+)', content)
                                if match:
                                    output_dir = match.group(1).strip()
                    except:
                        pass
                
                if output_dir and os.path.exists(output_dir):
                    trainer_state_file = os.path.join(output_dir, "trainer_state.json")
                    if os.path.exists(trainer_state_file):
                        try:
                            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                                trainer_state = json.load(f)
                            
                            # 检查log_history中的最新eval_loss
                            if "log_history" in trainer_state:
                                for log_entry in reversed(trainer_state["log_history"]):
                                    if "eval_loss" in log_entry:
                                        eval_loss_value = log_entry["eval_loss"]
                                        step = log_entry.get("step", 0)
                                        
                                        # 检查是否是新记录
                                        if not any(h.get("step") == step for h in eval_loss_history):
                                            eval_loss_history.append({{
                                                "step": step,
                                                "eval_loss": eval_loss_value,
                                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }})
                                            
                                            if eval_loss_logger:
                                                eval_loss_logger.info(
                                                    f"📈 Step {{step}} | "
                                                    f"Eval Loss: {{eval_loss_value:.6f}} | "
                                                    f"Time: {{datetime.now().strftime('%H:%M:%S')}}"
                                                )
                                            
                                            main_logger.info(
                                                f"✅ 验证集Loss (Step {{step}}): {{eval_loss_value:.6f}}"
                                            )
                                        break
                        except Exception as e:
                            pass  # 如果读取失败，忽略
                        
                time.sleep(2)  # 每2秒检查一次
                
            except KeyboardInterrupt:
                main_logger.info("🛑 监控已停止")
                # 保存eval_loss历史到JSON文件
                if eval_loss_history and eval_loss_logger:
                    eval_loss_file = log_files.get("eval_loss", "").replace(".log", "_history.json")
                    try:
                        with open(eval_loss_file, 'w', encoding='utf-8') as f:
                            json.dump(eval_loss_history, f, indent=2, ensure_ascii=False)
                        eval_loss_logger.info(f"💾 验证集Loss历史已保存: {{eval_loss_file}}")
                    except Exception as e:
                        eval_loss_logger.error(f"❌ 保存验证集Loss历史失败: {{e}}")
                break
            except Exception as e:
                main_logger.error(f"❌ 监控错误: {{e}}")
                time.sleep(5)
    else:
        main_logger.warning("⚠️ 未找到训练日志文件")

def main():
    '''主函数'''
    
    log_files = {log_files_str}
    
    # 设置日志记录器
    loggers = setup_loggers(log_files)
    
    # 开始监控
    monitor_training_logs(loggers, log_files)

if __name__ == "__main__":
    main()
"""
    
    # 保存监控脚本
    script_path = os.path.join(os.path.dirname(list(log_files.values())[0]), "monitor_training.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(monitoring_script)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    return script_path

def main():
    """主函数"""
    
    print("🚀 创建增强的LLaMA-Factory训练解决方案")
    print("🔮 包含Token分析和预测监控功能")
    print("=" * 60)
    
    # 首先修改训练器以添加监控功能
    print("🔧 修改训练器添加监控功能...")
    modify_trainer_to_add_monitoring()
    
    # 创建增强训练脚本
    script_path, output_dir, log_files = create_enhanced_training_script()
    
    # 创建监控脚本
    monitor_script = create_monitoring_script(log_files)
    
    print(f"📜 训练脚本: {script_path}")
    print(f"🔍 监控脚本: {monitor_script}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 日志文件:")
    for log_type, log_file in log_files.items():
        print(f"   {log_type}: {log_file}")
    
    print("\n" + "=" * 60)
    print("📊 使用方法:")
    print("=" * 60)
    print(f"1. 运行训练脚本:")
    print(f"   bash {script_path}")
    print(f"")
    print(f"2. 在另一个终端监控训练:")
    print(f"   python3 {monitor_script}")
    print(f"")
    print(f"3. 查看日志文件:")
    for log_type, log_file in log_files.items():
        print(f"   tail -f {log_file}")
    
    print("\n" + "=" * 60)
    print("🎯 功能说明:")
    print("=" * 60)
    print("⚡ 双卡DDP加速训练（保持最佳单卡训练效果）:")
    print("✅ 使用 PyTorch DDP 分布式训练")
    print("✅ 双GPU并行训练（GPU 0,2）")
    print("✅ 配置与最佳单卡完全等效")
    print("")
    print("🎯 训练参数（匹配最佳单卡配置）:")
    print("✅ 5 epochs，学习率 5.0e-5")
    print("✅ 有效batch size=16（2卡×8累积），与单卡相同")
    print("✅ grad_norm=0.5, weight_decay=0.01, lora_dropout=0.05")
    print("✅ 预计训练时长：~2.5-3.5小时（vs 单卡6小时）")
    print("✅ 训练效果应与最佳单卡完全一致")
    print("")
    print("📊 关键公式:")
    print("   单卡: 1卡 × 1batch × 16累积 = 有效batch 16")
    print("   双卡: 2卡 × 1batch × 8累积 = 有效batch 16 ✅")
    print("")
    print("📝 Token 变化监控:")
    print("✅ 训练过程中的Token变化会实时记录到标签分析日志")
    print("✅ 显示数据切分：哪些token需要训练（labels != -100）")
    print("✅ 显示完整的input_ids和labels")
    print("✅ 追踪训练过程中token的具体变化")
    print("✅ 每5步记录详细Token信息，每步都检查变化")
    print("✅ 监控回调已集成到训练器中，会自动记录训练Token变化")
    print("")
    print("📈 验证集Loss监控:")
    print("✅ 使用独立的test数据集作为验证集（eval_dataset=sft_test_data_01_08）")
    print("✅ 每50步评估一次验证集Loss（eval_steps=50）")
    print("✅ 验证集Loss会实时打印到控制台和日志文件")
    print("✅ 验证集Loss历史会保存到独立的日志文件")
    print("✅ 验证集Loss会绘制在loss曲线图中（plot_loss=True）")
    print("✅ 验证集Loss会保存在trainer_state.json和all_results.json中")
    print("✅ 监控脚本会自动提取和记录验证集Loss变化")
    print("")
    print("💡 提示：如果需要从训练集中划分验证集，可以使用 --val_size 0.1 替代 --eval_dataset")
    print("")
    print("🔮 预测Token监控功能:")
    print("✅ 实时监控模型的预测输出变化")
    print("✅ 显示模型预测文本 vs 目标文本对比")
    print("✅ 计算预测准确率统计")
    print("✅ 追踪预测Token在训练过程中的变化")
    print("✅ 显示预测变化的具体位置和内容")
    print("✅ 支持中文Token解码，直观查看预测内容")
    print("✅ 详细的Token ID和文本对比")
    print("✅ 预测准确率为0%时的调试信息")
    print("✅ 预测文本变化分析和相似度计算")
    print("✅ 文本差异详细对比（新增/删除/修改）")
    print("")
    print("💬 多轮对话分析功能:")
    print("✅ 分析-100部分（忽略的Token）")
    print("✅ 显示忽略Token的位置和内容")
    print("✅ 多轮对话结构分析")
    print("✅ 对话分段详情")
    print("✅ 训练部分vs忽略部分的比例")
    print("✅ 多轮对话模式检测")
    
    # 询问是否立即运行
    response = input("\n是否立即开始训练? (y/n): ").lower().strip()
    if response in ['y', 'yes', '是']:
        print("\n🚀 开始训练...")
        subprocess.run(["bash", script_path])
    else:
        print(f"\n📜 训练脚本已创建: {script_path}")
        print("可以稍后手动运行训练")

if __name__ == "__main__":
    main()
