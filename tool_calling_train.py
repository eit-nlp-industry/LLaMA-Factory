#!/usr/bin/env python3
"""
完整的增强训练启动脚本
整合所有优化功能，一键启动训练

使用方法：
1. 在脚本顶部修改超参数配置
2. 直接运行：python run_enhanced_training_complete.py
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================================
# 超参数配置区域 - 在这里直接修改训练参数
# ============================================================================
#
# ⚠️ 过拟合实验配置 ⚠️
# 本配置专门用于学习过拟合现象和各参数的作用
# 预期效果：训练集指标会非常好看（0.95-0.99），但验证集指标可能下降
# 目的：通过实验理解过拟合的含义和影响
#
# 如需恢复正常配置，请从 backups/ 目录恢复备份文件
# ============================================================================

# 模型和数据集配置
MODEL_PATH = "/data/models/Qwen3-8B"  # 模型路径
DATASET_NAME = "tool_calling_12_08"  # 数据集名称（在dataset_info.json中定义）
TEST_DATASET_NAME = "tool_calling_12_08_test"  # 测试数据集名称
TRAIN_DATA_PATH = "data/dataset/12_08/train.json"  # 训练数据路径
TEST_DATA_PATH = "data/dataset/12_08/test.json"  # 测试数据路径

# 学习率和训练配置（过拟合优化）
LEARNING_RATE = 5e-6  # 学习率（从1e-5降低到5e-6，更精细拟合，便于记忆训练数据）
NUM_TRAIN_EPOCHS = 20.0  # 训练轮数（从12增加到20，充分过拟合训练集）
MAX_SAMPLES = 100000  # 最大样本数

# 批次配置（过拟合优化）
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 单设备批次大小
GRADIENT_ACCUMULATION_STEPS = 32  # 梯度累积步数（从16增加到32，更频繁的梯度更新，有效batch=1×32）
PER_DEVICE_EVAL_BATCH_SIZE = 1  # 评估批次大小（建议设为1以节省评估时的内存）
LR_SCHEDULER_TYPE = "constant"  # 学习率调度器类型（从"cosine"改为"constant"，保持恒定学习率，不衰减）
WARMUP_RATIO = 0.0  # Warmup比例（从0.05改为0.0，关闭warmup，立即开始学习）

# 正则化和稳定性（过拟合优化：关闭所有正则化）
MAX_GRAD_NORM = 10.0  # 梯度裁剪阈值（从0.5增加到10.0，几乎不裁剪梯度，允许更大的参数更新）
WEIGHT_DECAY = 0.0  # 权重衰减（从0.01改为0.0，关闭权重衰减，允许模型自由记忆）

# LoRA配置（过拟合优化：提高模型容量）
LORA_RANK = 128  # LoRA rank（从32增加到128，提高模型表达能力，便于记忆更多训练数据）
LORA_ALPHA = 256  # LoRA alpha（从64增加到256，保持2倍关系，增强LoRA的影响）
LORA_DROPOUT = 0.0  # LoRA dropout（从0.05改为0.0，完全关闭dropout，允许模型完全记忆）
# 仅挂核心注意力/MLP，减少噪声、集中学习
LORA_TARGET = "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

# 训练设置
CUTOFF_LEN = 8192  # 序列最大长度（如果OOM，可尝试降低到4096或2048）
LOGGING_STEPS = 10  # 日志记录步数
SAVE_STEPS = 20  # 模型保存步数（从500改为20，确保能捕获最佳模型）
EVAL_STEPS = 20  # 评估步数（从500改为20，更频繁评估以监控过拟合）
SAVE_TOTAL_LIMIT = 10  # 保留的checkpoint数量（从3增加到10，保存更多历史checkpoint）

# 最佳模型保存配置
LOAD_BEST_MODEL_AT_END = True  # 训练结束时加载最佳模型
METRIC_FOR_BEST_MODEL = "eval_loss"  # 用于选择最佳模型的指标（eval_loss越小越好）
GREATER_IS_BETTER = False  # eval_loss越小越好，所以设为False

# 其他配置
TEMPLATE = "qwen3"  # 模板类型
FINETUNING_TYPE = "lora"  # 微调类型
PREPROCESSING_NUM_WORKERS = 16  # 数据预处理工作进程数
DATALOADER_NUM_WORKERS = 4  # 数据加载工作进程数
FLASH_ATTN = "auto"  # Flash attention设置
GRADIENT_CHECKPOINTING = True  # 是否启用梯度检查点
BF16 = True  # 是否使用bf16精度
OPTIMIZER = "adamw_torch"  # 优化器: "adamw_torch" 或 "adamw_bnb_8bit" (8-bit优化器可节省内存)

# 内存优化配置
# 方案1: 使用 DeepSpeed ZeRO-3 (推荐，内存节省最多，但需要安装 deepspeed)
USE_DEEPSPEED = True  # 是否启用 DeepSpeed ZeRO-3
DEEPSPEED_CONFIG = "examples/deepspeed/ds_z3_config.json"  # DeepSpeed 配置文件路径

# 方案2: 使用 8-bit 优化器 (简单，但节省较少，需要安装 bitsandbytes)
# 如果 USE_DEEPSPEED=False，可以设置 OPTIMIZER="adamw_bnb_8bit" 来启用 8-bit 优化器

# CUDA配置
CUDA_VISIBLE_DEVICES = "6"  # 使用的GPU设备，单卡训练使用 "6"

# 自动执行选项
AUTO_VALIDATE_DATA = True  # 自动验证数据
AUTO_ENHANCE_DATA = True  # 自动增强数据（应用增强的系统提示）
SKIP_IF_ENHANCED_EXISTS = True  # 如果增强数据已存在则跳过

# 备份配置（便于回退/对比）
BACKUP_BASE_CONFIG = {
    "GRADIENT_ACCUMULATION_STEPS": 16,
    "OPTIMIZER": "adamw_bnb_8bit",
    "LEARNING_RATE": 2e-5,
    "LORA_DROPOUT": 0.02,
    "LORA_TARGET": "all",
    "NUM_TRAIN_EPOCHS": 3.0,
}

# ============================================================================
# 以下为脚本逻辑，通常不需要修改
# ============================================================================

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查必要文件（使用配置的路径）
    required_files = [
        TRAIN_DATA_PATH,
        TEST_DATA_PATH,
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
    if not AUTO_VALIDATE_DATA:
        print("\\n⏭️  跳过数据验证（AUTO_VALIDATE_DATA=False）")
        return True
    
    print("\\n🔍 验证训练数据...")
    
    validator_path = "validate_tool_calling_data.py"
    setup_script = "tool_calling_setup.py"
    
    # 强制重新生成验证脚本，确保使用最新版本
    if Path(setup_script).exists():
        try:
            # 导入并调用create_data_validator函数
            import importlib.util
            spec = importlib.util.spec_from_file_location("tool_calling_setup", setup_script)
            setup_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(setup_module)
            setup_module.create_data_validator()
            if Path(validator_path).exists():
                print("✅ 验证工具已更新到最新版本")
            else:
                print("✅ 验证工具已重新生成")
        except Exception as e:
            print(f"⚠️  重新生成验证工具失败: {e}")
            if not Path(validator_path).exists():
                print("⚠️  验证工具不存在，跳过验证")
                return True
    elif not Path(validator_path).exists():
        print("⚠️  验证工具不存在且无法重新生成，跳过验证")
        return True
    
    try:
        result = subprocess.run(
            [sys.executable, validator_path, TRAIN_DATA_PATH],
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
    if not AUTO_ENHANCE_DATA:
        print("\\n⏭️  跳过数据增强（AUTO_ENHANCE_DATA=False）")
        return TRAIN_DATA_PATH
    
    print("\\n🔧 检查数据增强...")
    
    enhanced_path = str(Path(TRAIN_DATA_PATH).parent / "train_enhanced.json")
    
    if Path(enhanced_path).exists() and SKIP_IF_ENHANCED_EXISTS:
        print(f"✅ 增强数据已存在: {enhanced_path}")
        update_dataset_info_for_enhanced(enhanced_path)
        return enhanced_path
    
    enhancer_path = "tool_calling_enhance_data.py"
    if not Path(enhancer_path).exists():
        print("⚠️  数据增强工具不存在，使用原始数据")
        return TRAIN_DATA_PATH
    
    print("📝 开始增强数据...")
    try:
        subprocess.run(
            [sys.executable, enhancer_path, 
             TRAIN_DATA_PATH, 
             enhanced_path],
            check=True
        )
        print(f"✅ 数据增强完成: {enhanced_path}")
        
        # 更新dataset_info.json使用增强数据
        update_dataset_info_for_enhanced(enhanced_path)
        return enhanced_path
    except Exception as e:
        print(f"⚠️  数据增强失败: {e}，使用原始数据")
        return TRAIN_DATA_PATH

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

def create_deepspeed_config(config_path):
    """创建 DeepSpeed ZeRO-3 配置文件"""
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(ds_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已创建 DeepSpeed ZeRO-3 配置文件: {config_path}")

def create_training_command(output_dir=None, model_path=None):
    """创建训练命令，使用脚本顶部的超参数配置"""
    
    # 使用全局配置或参数
    model_path = model_path or MODEL_PATH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = output_dir or f"saves/Qwen3-8B/lora/enhanced_tool_calling_{timestamp}"
    
    # 构建训练命令，使用脚本顶部的配置
    cmd = [
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--do_train", "True",
        "--model_name_or_path", model_path,
        "--preprocessing_num_workers", str(PREPROCESSING_NUM_WORKERS),
        "--finetuning_type", FINETUNING_TYPE,
        "--template", TEMPLATE,
        "--flash_attn", FLASH_ATTN,
        "--dataset_dir", "data",
        "--dataset", DATASET_NAME,
        "--cutoff_len", str(CUTOFF_LEN),
        
        # 学习率和训练轮数
        "--learning_rate", str(LEARNING_RATE),
        "--num_train_epochs", str(NUM_TRAIN_EPOCHS),
        "--max_samples", str(MAX_SAMPLES),
        
        # 批次配置
        "--per_device_train_batch_size", str(PER_DEVICE_TRAIN_BATCH_SIZE),
        "--per_device_eval_batch_size", str(PER_DEVICE_EVAL_BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
        "--lr_scheduler_type", LR_SCHEDULER_TYPE,
        "--warmup_ratio", str(WARMUP_RATIO),
        
        # 正则化和稳定性
        "--max_grad_norm", str(MAX_GRAD_NORM),
        "--weight_decay", str(WEIGHT_DECAY),
        "--lora_rank", str(LORA_RANK),
        "--lora_alpha", str(LORA_ALPHA),
        "--lora_dropout", str(LORA_DROPOUT),
        
        # 训练设置
        "--logging_steps", str(LOGGING_STEPS),
        "--save_steps", str(SAVE_STEPS),
        "--save_strategy", "steps",
        "--eval_strategy", "steps",
        "--eval_steps", str(EVAL_STEPS),
        "--eval_dataset", TEST_DATASET_NAME,
        "--packing", "False",
        "--enable_thinking", "False",
        "--overwrite_cache", "True",
        
        # 最佳模型保存配置
        "--load_best_model_at_end", str(LOAD_BEST_MODEL_AT_END),
        "--metric_for_best_model", METRIC_FOR_BEST_MODEL,
        "--greater_is_better", str(GREATER_IS_BETTER),
        
        # 输出
        "--output_dir", final_output_dir,
        "--bf16", str(BF16),
        "--plot_loss", "True",
        "--trust_remote_code", "True",
        "--ddp_timeout", "180000000",
        "--include_num_input_tokens_seen", "True",
        "--optim", OPTIMIZER,
        "--lora_target", LORA_TARGET,
        "--gradient_checkpointing", str(GRADIENT_CHECKPOINTING),
        
        # 数据加载
        "--dataloader_pin_memory", "False",
        "--dataloader_num_workers", str(DATALOADER_NUM_WORKERS),
        "--remove_unused_columns", "False",
        "--dataloader_drop_last", "False",
        
        # 其他
        "--seed", "42",
        "--save_total_limit", str(SAVE_TOTAL_LIMIT),
    ]
    
    # 添加 DeepSpeed 配置（如果启用）
    if USE_DEEPSPEED:
        # 检查配置文件是否存在
        if not Path(DEEPSPEED_CONFIG).exists():
            print(f"⚠️  DeepSpeed 配置文件不存在: {DEEPSPEED_CONFIG}")
            print(f"   将使用默认的 DeepSpeed ZeRO-3 配置")
            # 创建默认的 DeepSpeed ZeRO-3 配置
            create_deepspeed_config(DEEPSPEED_CONFIG)
        cmd.append("--deepspeed")
        cmd.append(DEEPSPEED_CONFIG)
    
    return cmd, final_output_dir

def main():
    """主函数 - 自动执行全部步骤"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="增强的工具调用训练启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用说明：
1. 在脚本顶部修改超参数配置（MODEL_PATH, LEARNING_RATE等）
2. 直接运行：python tool_calling_train.py
3. 脚本会自动完成：环境检查 → 数据验证 → 数据增强 → 训练执行

或者使用命令行参数覆盖配置：
  --model_path: 覆盖MODEL_PATH
  --output_dir: 指定输出目录
  --skip_validation: 跳过数据验证
  --skip_enhancement: 跳过数据增强
  --dry_run: 只显示命令，不执行
        """
    )
    parser.add_argument("--model_path", type=str, default=None, 
                       help=f"模型路径（覆盖脚本中的MODEL_PATH，默认: {MODEL_PATH}）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认: 自动生成）")
    parser.add_argument("--skip_validation", action="store_true",
                       help="跳过数据验证（覆盖AUTO_VALIDATE_DATA）")
    parser.add_argument("--skip_enhancement", action="store_true",
                       help="跳过数据增强（覆盖AUTO_ENHANCE_DATA）")
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示命令，不执行")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="继续训练的checkpoint路径，透传给llamafactory-cli")
    
    args = parser.parse_args()
    
    print("🚀 增强的工具调用训练启动脚本")
    print("=" * 60)
    print("📝 提示: 在脚本顶部修改超参数配置（MODEL_PATH, LEARNING_RATE等）")
    print("=" * 60)
    
    # 1. 检查环境
    if not check_environment():
        print("\\n❌ 环境检查失败，请修复后重试")
        sys.exit(1)
    
    # 2. 验证数据（根据配置和参数）
    if not args.skip_validation:
        validate_data()
    else:
        print("\\n⏭️  跳过数据验证（--skip_validation）")
    
    # 3. 增强数据（根据配置和参数）
    if not args.skip_enhancement:
        data_path = enhance_data_if_needed()
    else:
        data_path = TRAIN_DATA_PATH
        print(f"\\n⏭️  跳过数据增强（--skip_enhancement）")
        print(f"📝 使用原始数据: {data_path}")
    
    # 4. 创建训练命令（使用脚本顶部的配置）
    print("\\n⚙️  准备训练命令...")
    cmd, output_dir = create_training_command(args.output_dir, args.model_path)
    if args.resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", args.resume_from_checkpoint]
    
    print(f"\\n📊 训练配置（来自脚本顶部配置）:")
    print(f"   模型路径: {args.model_path or MODEL_PATH}")
    print(f"   输出目录: {output_dir}")
    print(f"   数据集: {DATASET_NAME}")
    print(f"   学习率: {LEARNING_RATE} (降低，更精细拟合)")
    print(f"   训练轮数: {NUM_TRAIN_EPOCHS} (增加，充分过拟合)")
    print(f"   LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT} (提高容量，关闭正则化)")
    print(f"   有效batch size: {PER_DEVICE_TRAIN_BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} (增加，更频繁更新)")
    print(f"   梯度裁剪: {MAX_GRAD_NORM} (放宽，几乎不裁剪)")
    print(f"   权重衰减: {WEIGHT_DECAY} (关闭，允许自由记忆)")
    print(f"   学习率调度: {LR_SCHEDULER_TYPE} (恒定，不衰减)")
    print(f"   Warmup比例: {WARMUP_RATIO} (关闭)")
    print(f"   保存频率: 每 {SAVE_STEPS} 步保存一次")
    print(f"   评估频率: 每 {EVAL_STEPS} 步评估一次")
    print(f"   最佳模型保存: {LOAD_BEST_MODEL_AT_END} (基于 {METRIC_FOR_BEST_MODEL})")
    print(f"   保留checkpoint数: {SAVE_TOTAL_LIMIT}")
    if USE_DEEPSPEED:
        print(f"   DeepSpeed ZeRO-3: 已启用 (配置文件: {DEEPSPEED_CONFIG})")
    if OPTIMIZER == "adamw_bnb_8bit":
        print(f"   8-bit 优化器: 已启用")
    print(f"\\n   💡 预期效果:")
    print(f"      - 训练集指标: 0.95-0.99 (非常好看)")
    print(f"      - 验证集指标: 可能下降 (过拟合)")
    print(f"      - 目的: 学习过拟合的含义和各参数的作用")
    
    if args.dry_run:
        print(f"\\n📜 训练命令（dry-run模式）:")
        print(" ".join(cmd))
        return
    
    # 5. 设置CUDA设备
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        print(f"\\n🎮 设置CUDA设备: {CUDA_VISIBLE_DEVICES}")
    
    # 5.5. 如果启用 DeepSpeed，设置 FORCE_TORCHRUN（DeepSpeed 需要 torchrun）
    if USE_DEEPSPEED:
        os.environ["FORCE_TORCHRUN"] = "1"
        print(f"\\n⚙️  启用 DeepSpeed，已设置 FORCE_TORCHRUN=1")
    
    # 6. 执行训练
    print(f"\\n🚀 开始训练...")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\\n" + "=" * 60)
        print("✅ 训练完成！")
        print(f"📁 模型保存在: {output_dir}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\\n⚠️  训练被用户中断")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ 训练失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

