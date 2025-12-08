#!/usr/bin/env python3
"""
完整的增强训练启动脚本
整合所有优化功能，一键启动训练
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查必要文件
    required_files = [
        "data/dataset/12_08/train.json",
        "data/dataset/12_08/test.json",
        "data/dataset_info.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少必要文件:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("✅ 环境检查通过")
    return True

def validate_data():
    """验证训练数据"""
    print("\\n🔍 验证训练数据...")
    
    validator_path = "validate_tool_calling_data.py"
    if not Path(validator_path).exists():
        print(f"⚠️  验证工具不存在，跳过验证")
        return True
    
    try:
        result = subprocess.run(
            [sys.executable, validator_path, "data/dataset/12_08/train.json"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print("⚠️  数据验证发现问题，但将继续训练")
        return True
    except Exception as e:
        print(f"⚠️  验证过程出错: {e}")
        return True

def enhance_data_if_needed():
    """如果需要，增强数据"""
    print("\\n🔧 检查数据增强...")
    
    enhanced_path = "data/dataset/12_08/train_enhanced.json"
    
    if Path(enhanced_path).exists():
        print(f"✅ 增强数据已存在: {enhanced_path}")
        return enhanced_path
    
    enhancer_path = "enhance_dataset_with_constraints.py"
    if not Path(enhancer_path).exists():
        print("⚠️  数据增强工具不存在，使用原始数据")
        return "data/dataset/12_08/train.json"
    
    print("📝 开始增强数据...")
    try:
        subprocess.run(
            [sys.executable, enhancer_path, 
             "data/dataset/12_08/train.json", 
             enhanced_path],
            check=True
        )
        print(f"✅ 数据增强完成: {enhanced_path}")
        
        # 更新dataset_info.json使用增强数据
        update_dataset_info_for_enhanced(enhanced_path)
        return enhanced_path
    except Exception as e:
        print(f"⚠️  数据增强失败: {e}，使用原始数据")
        return "data/dataset/12_08/train.json"

def update_dataset_info_for_enhanced(enhanced_path):
    """更新dataset_info.json以使用增强数据"""
    dataset_info_path = "data/dataset_info.json"
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # 更新路径为相对路径
    rel_path = str(Path(enhanced_path).relative_to("data"))
    dataset_info["tool_calling_12_08"]["file_name"] = rel_path
    
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已更新 {dataset_info_path}")

def create_training_command(output_dir, model_path=None):
    """创建训练命令"""
    
    if model_path is None:
        model_path = "/data/models/Qwen3-8B"  # 默认路径，需要根据实际情况修改
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = f"saves/Qwen3-8B/lora/enhanced_tool_calling_{timestamp}"
    
    if output_dir:
        final_output_dir = output_dir
    
    # 优化的超参数配置
    cmd = [
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--do_train", "True",
        "--model_name_or_path", model_path,
        "--preprocessing_num_workers", "16",
        "--finetuning_type", "lora",
        "--template", "qwen3",
        "--flash_attn", "auto",
        "--dataset_dir", "data",
        "--dataset", "tool_calling_12_08",
        "--cutoff_len", "8192",
        
        # 学习率和训练轮数
        "--learning_rate", "2.0e-5",
        "--num_train_epochs", "8.0",
        "--max_samples", "100000",
        
        # 批次配置
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.1",
        
        # 正则化和稳定性
        "--max_grad_norm", "0.3",
        "--weight_decay", "0.01",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--lora_dropout", "0.1",
        
        # 训练设置
        "--logging_steps", "10",
        "--save_steps", "500",
        "--save_strategy", "steps",
        "--evaluation_strategy", "steps",
        "--eval_steps", "500",
        "--eval_dataset", "tool_calling_12_08_test",
        "--packing", "False",
        "--enable_thinking", "False",
        "--overwrite_cache", "True",
        
        # 输出
        "--output_dir", final_output_dir,
        "--bf16", "True",
        "--plot_loss", "True",
        "--trust_remote_code", "True",
        "--ddp_timeout", "180000000",
        "--include_num_input_tokens_seen", "True",
        "--optim", "adamw_torch",
        "--lora_target", "all",
        "--gradient_checkpointing", "True",
        
        # 数据加载
        "--dataloader_pin_memory", "False",
        "--dataloader_num_workers", "4",
        "--remove_unused_columns", "False",
        "--dataloader_drop_last", "False",
        
        # 其他
        "--seed", "42",
        "--save_total_limit", "3",
    ]
    
    return cmd, final_output_dir

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强的工具调用训练启动脚本")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="模型路径（默认: /data/models/Qwen3-8B）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认: 自动生成）")
    parser.add_argument("--skip_validation", action="store_true",
                       help="跳过数据验证")
    parser.add_argument("--skip_enhancement", action="store_true",
                       help="跳过数据增强")
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示命令，不执行")
    
    args = parser.parse_args()
    
    print("🚀 增强的工具调用训练启动脚本")
    print("=" * 60)
    
    # 1. 检查环境
    if not check_environment():
        print("\\n❌ 环境检查失败，请修复后重试")
        sys.exit(1)
    
    # 2. 验证数据
    if not args.skip_validation:
        validate_data()
    
    # 3. 增强数据
    if not args.skip_enhancement:
        data_path = enhance_data_if_needed()
    else:
        data_path = "data/dataset/12_08/train.json"
        print(f"\\n📝 使用原始数据: {data_path}")
    
    # 4. 创建训练命令
    print("\\n⚙️  准备训练命令...")
    cmd, output_dir = create_training_command(args.output_dir, args.model_path)
    
    print(f"\\n📊 训练配置:")
    print(f"   模型路径: {args.model_path or '/data/models/Qwen3-8B'}")
    print(f"   输出目录: {output_dir}")
    print(f"   数据集: tool_calling_12_08")
    print(f"   学习率: 2.0e-5")
    print(f"   训练轮数: 8.0")
    print(f"   LoRA rank: 64, alpha: 128")
    print(f"   有效batch size: 16 (1 × 16)")
    
    if args.dry_run:
        print(f"\\n📜 训练命令（dry-run模式）:")
        print(" ".join(cmd))
        return
    
    # 5. 执行训练
    print(f"\\n🚀 开始训练...")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\\n" + "=" * 60)
        print("✅ 训练完成！")
        print(f"📁 模型保存在: {output_dir}")
    except KeyboardInterrupt:
        print("\\n⚠️  训练被用户中断")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ 训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

