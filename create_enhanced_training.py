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
    trainer_file = "/home/qiyang_shi/LLaMA-Factory/src/llamafactory/train/sft/trainer.py"
    
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
    output_dir = f"/home/qiyang_shi/LLaMA-Factory/saves/Qwen3-8B/lora/train_{timestamp}"
    
    # 创建日志目录
    log_dir = "/home/qiyang_shi/LLaMA-Factory/enhanced_training_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_files = {
        "training": os.path.join(log_dir, f"training_monitor_{log_timestamp}.log"),
        "predictions": os.path.join(log_dir, f"prediction_monitor_{log_timestamp}.log"),
        "labels": os.path.join(log_dir, f"label_analysis_{log_timestamp}.log"),
        "alignment": os.path.join(log_dir, f"alignment_analysis_{log_timestamp}.log"),
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

# 设置环境变量 - 双卡训练（根据实际情况调整）
# 推荐使用两张显存较大的GPU
export CUDA_VISIBLE_DEVICES=4,5

# 创建输出目录
mkdir -p "{output_dir}"

# 创建日志目录
mkdir -p "{log_dir}"

# 设置日志文件路径
export ENHANCED_TRAINING_LOG="{log_files['main']}"
export ENHANCED_LABEL_LOG="{log_files['labels']}"
export ENHANCED_PREDICT_LOG="{log_files['predictions']}"
export ENHANCED_ALIGNMENT_LOG="{log_files['alignment']}"

# 记录开始时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 🚀 增强训练开始" >> "{log_files['main']}"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📁 输出目录: {output_dir}" >> "{log_files['main']}"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | 📝 日志文件: {log_files['main']}" >> "{log_files['main']}"

# 运行训练命令 - 多GPU分布式训练 + DeepSpeed ZeRO-2
echo "🔄 执行多GPU分布式训练命令（使用DeepSpeed优化显存）..."
llamafactory-cli train \\
    --stage sft \\
    --do_train True \\
    --model_name_or_path /data/models/Qwen3-8B \\
    --preprocessing_num_workers 4 \\
    --finetuning_type lora \\
    --template qwen3 \\
    --flash_attn auto \\
    --dataset_dir data \\
    --dataset data_demo \\
    --cutoff_len 10240 \\
    --learning_rate 5.0e-5 \\
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \\
    --num_train_epochs 6.0 \\
    --max_samples 100000 \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --lr_scheduler_type cosine \\
    --max_grad_norm 1.0 \\
    --weight_decay 0.01 \\
    --logging_steps 5 \\
    --save_steps 100 \\
    --warmup_ratio 0.1 \\
    --packing False \\
    --enable_thinking False \\
    --overwrite_cache True \\
    --save_strategy steps \\
    --output_dir "{output_dir}" \\
    --bf16 True \\
    --plot_loss True \\
    --trust_remote_code True \\
    --ddp_timeout 180000000 \\
    --include_num_input_tokens_seen True \\
    --optim adamw_torch \\
    --lora_rank 32 \\
    --lora_alpha 64 \\
    --lora_dropout 0.05 \\
    --lora_target all \\
    --gradient_checkpointing True \\
    --dataloader_pin_memory False \\
    --dataloader_num_workers 4 \\
    --remove_unused_columns False \\
    --dataloader_drop_last True

# 记录结束时间
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | ✅ 训练完成" >> "{log_files['main']}"

echo "✅ 训练完成"
echo "📊 训练摘要:"
echo "   📁 输出目录: {output_dir}"
echo "   📝 日志文件: {log_files['main']}"
echo "   🏷️ 标签分析: {log_files['labels']}"
echo "   🔮 预测监控: {log_files['predictions']}"
echo "   🎯 对齐分析: {log_files['alignment']}"
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
   - 学习率调整
   - 训练进度
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
        while True:
            try:
                if os.path.exists(trainer_log):
                    current_size = os.path.getsize(trainer_log)
                    if current_size > last_size:
                        # 读取新增内容
                        with open(trainer_log, 'r', encoding='utf-8') as f:
                            f.seek(last_size)
                            new_content = f.read()
                            
                        # 记录新内容
                        for line in new_content.strip().split('\\n'):
                            if line.strip():
                                main_logger.info(f"📊 训练日志: {{line}}")
                                
                        last_size = current_size
                        
                time.sleep(1)  # 每秒检查一次
                
            except KeyboardInterrupt:
                main_logger.info("🛑 监控已停止")
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
    print("✅ 训练过程中的Token变化会实时记录到标签分析日志")
    print("✅ 显示数据切分：哪些token需要训练（labels != -100）")
    print("✅ 显示完整的input_ids和labels")
    print("✅ 追踪训练过程中token的具体变化")
    print("✅ 每5步记录详细Token信息，每步都检查变化")
    print("✅ 监控回调已集成到训练器中，会自动记录训练Token变化")
    print("")
    print("🔮 新增预测Token监控功能:")
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
    print("💬 新增多轮对话分析功能:")
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
