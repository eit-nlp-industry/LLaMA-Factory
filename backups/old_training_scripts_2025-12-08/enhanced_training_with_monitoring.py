#!/usr/bin/env python3
"""
增强的训练脚本 - 集成Label打印、Predict监控和对齐分析
基于LLaMA-Factory的原始训练流程，添加详细的监控和日志记录功能
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# 添加LLaMA-Factory路径
sys.path.insert(0, "/home/ziqiang/LLaMA-Factory/src")

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import PredictionOutput

# 导入LLaMA-Factory组件
from llamafactory.hparams import (
    DataArguments,
    FinetuningArguments, 
    ModelArguments,
    GeneratingArguments
)
from llamafactory.data import get_dataset
from llamafactory.model import get_model_and_tokenizer
from llamafactory.train.sft import get_trainer
from llamafactory.extras import logging as llamafactory_logging

# 导入自定义组件
from predict_monitoring_callback import PredictMonitoringCallback
from enhanced_label_debug import EnhancedLabelDebugger

class EnhancedTrainingMonitor:
    """增强的训练监控器"""
    
    def __init__(self, 
                 model_name: str,
                 output_dir: str,
                 log_interval: int = 10,
                 detailed_analysis: bool = True):
        """
        初始化训练监控器
        
        Args:
            model_name: 模型名称
            output_dir: 输出目录
            log_interval: 日志记录间隔
            detailed_analysis: 是否进行详细分析
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.detailed_analysis = detailed_analysis
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化调试器
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(output_dir, f"enhanced_training_monitor_{timestamp}.log")
        
        self.debugger = EnhancedLabelDebugger(
            model_name=model_name,
            log_file=self.log_file
        )
        
        # 存储训练数据
        self.training_data = {
            "steps": [],
            "losses": [],
            "learning_rates": [],
            "predictions": [],
            "labels": [],
            "alignment_scores": []
        }
        
        self.debugger.log_debug("🚀 增强训练监控器初始化完成")
        self.debugger.log_debug(f"📁 输出目录: {output_dir}")
        self.debugger.log_debug(f"📝 日志文件: {self.log_file}")
        self.debugger.log_debug(f"⏱️ 日志间隔: {log_interval}步")
    
    def log_training_step(self, step: int, loss: float, lr: float, **kwargs):
        """记录训练步骤"""
        self.training_data["steps"].append(step)
        self.training_data["losses"].append(loss)
        self.training_data["learning_rates"].append(lr)
        
        if step % self.log_interval == 0:
            self.debugger.log_debug(f"\n🔄 训练步骤 {step}")
            self.debugger.log_debug(f"{'=' * 50}")
            self.debugger.log_debug(f"📉 Loss: {loss:.6f}")
            self.debugger.log_debug(f"📊 学习率: {lr:.2e}")
            
            # 记录其他指标
            for key, value in kwargs.items():
                if value is not None:
                    self.debugger.log_debug(f"📈 {key}: {value}")
    
    def log_prediction_analysis(self, step: int, predictions: List[int], labels: List[int], loss: Optional[float] = None):
        """记录预测分析"""
        self.training_data["predictions"].append(predictions)
        self.training_data["labels"].append(labels)
        
        if self.detailed_analysis:
            analysis = self.debugger.analyze_training_step(
                step=step,
                predictions=predictions,
                labels=labels,
                loss=loss
            )
            
            alignment_score = analysis["alignment_analysis"]["valid_match_percentage"]
            self.training_data["alignment_scores"].append(alignment_score)
            
            self.debugger.log_debug(f"\n🔮 预测分析 - 步骤 {step}")
            self.debugger.log_debug(f"{'=' * 50}")
            self.debugger.log_debug(f"🎯 对齐分数: {alignment_score:.1f}%")
            self.debugger.log_debug(f"📊 有效匹配: {analysis['alignment_analysis']['valid_matches']}/{analysis['alignment_analysis']['total_valid_labels']}")
            
            if loss is not None:
                self.debugger.log_debug(f"📉 预测Loss: {loss:.6f}")
    
    def save_training_summary(self):
        """保存训练摘要"""
        summary_file = os.path.join(self.output_dir, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        summary = {
            "model_name": self.model_name,
            "training_data": self.training_data,
            "total_steps": len(self.training_data["steps"]),
            "final_loss": self.training_data["losses"][-1] if self.training_data["losses"] else None,
            "avg_alignment_score": sum(self.training_data["alignment_scores"]) / len(self.training_data["alignment_scores"]) if self.training_data["alignment_scores"] else None,
            "analysis_time": datetime.now().isoformat()
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.debugger.log_debug(f"💾 训练摘要已保存到: {summary_file}")
        return summary_file

class EnhancedTrainingCallback(TrainerCallback):
    """增强的训练回调"""
    
    def __init__(self, monitor: EnhancedTrainingMonitor):
        self.monitor = monitor
        self.step_count = 0
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练步骤结束回调"""
        self.step_count += 1
        
        # 获取当前指标
        current_log = state.log_history[-1] if state.log_history else {}
        loss = current_log.get("loss")
        lr = current_log.get("learning_rate")
        
        if loss is not None:
            self.monitor.log_training_step(
                step=self.step_count,
                loss=loss,
                lr=lr or 0.0,
                epoch=current_log.get("epoch"),
                gradient_norm=current_log.get("grad_norm")
            )
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """评估回调"""
        self.monitor.debugger.log_debug(f"\n📊 评估阶段 - 步骤 {self.step_count}")
        self.monitor.debugger.log_debug(f"{'=' * 50}")
        
        # 如果有预测结果，进行分析
        if "eval_dataloader" in kwargs:
            eval_dataloader = kwargs["eval_dataloader"]
            if hasattr(eval_dataloader, "__len__") and len(eval_dataloader) > 0:
                self.monitor.debugger.log_debug(f"📦 评估数据大小: {len(eval_dataloader)}")
    
    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """预测回调"""
        self.monitor.debugger.log_debug(f"\n🔮 预测阶段 - 步骤 {self.step_count}")
        self.monitor.debugger.log_debug(f"{'=' * 50}")
        
        # 获取预测结果
        predict_results = kwargs.get("predict_results")
        if predict_results is not None:
            self._analyze_predictions(predict_results)
    
    def _analyze_predictions(self, predict_results: PredictionOutput):
        """分析预测结果"""
        predictions = predict_results.predictions
        labels = predict_results.label_ids
        
        if predictions is None or labels is None:
            self.monitor.debugger.log_debug("⚠️ 预测结果或标签为空")
            return
        
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 分析前几个样本
        batch_size = min(len(predictions), 3)
        for i in range(batch_size):
            pred_sample = predictions[i]
            label_sample = labels[i]
            
            # 移除padding
            pred_sample = self._remove_padding(pred_sample)
            label_sample = self._remove_padding(label_sample)
            
            # 记录预测分析
            eval_loss = predict_results.metrics.get("eval_loss") if hasattr(predict_results, "metrics") else None
            self.monitor.log_prediction_analysis(
                step=self.step_count,
                predictions=pred_sample.tolist(),
                labels=label_sample.tolist(),
                loss=eval_loss
            )
    
    def _remove_padding(self, tokens, pad_token_id: int = None) -> List[int]:
        """移除padding tokens"""
        if pad_token_id is None:
            pad_token_id = self.monitor.debugger.tokenizer.pad_token_id
        
        # 找到非padding的位置
        non_pad_mask = tokens != pad_token_id
        if any(non_pad_mask):
            # 找到第一个和最后一个非padding位置
            first_non_pad = next(i for i, x in enumerate(non_pad_mask) if x)
            last_non_pad = len(tokens) - 1 - next(i for i, x in enumerate(reversed(non_pad_mask)) if x)
            return tokens[first_non_pad:last_non_pad+1].tolist()
        else:
            return tokens.tolist()
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束回调"""
        self.monitor.debugger.log_debug(f"\n🏁 训练结束")
        self.monitor.debugger.log_debug(f"{'=' * 50}")
        self.monitor.debugger.log_debug(f"📊 总训练步骤: {self.step_count}")
        
        # 保存训练摘要
        summary_file = self.monitor.save_training_summary()
        self.monitor.debugger.log_debug(f"✅ 训练完成，摘要已保存到: {summary_file}")

def create_enhanced_trainer(model_args: ModelArguments,
                           data_args: DataArguments,
                           training_args: TrainingArguments,
                           finetuning_args: FinetuningArguments,
                           generating_args: GeneratingArguments,
                           **kwargs) -> Trainer:
    """创建增强的训练器"""
    
    # 创建监控器
    monitor = EnhancedTrainingMonitor(
        model_name=model_args.model_name_or_path,
        output_dir=training_args.output_dir,
        log_interval=training_args.logging_steps,
        detailed_analysis=True
    )
    
    # 获取模型和分词器
    model, tokenizer = get_model_and_tokenizer(model_args, finetuning_args)
    
    # 获取数据集
    dataset = get_dataset(model_args, data_args, training_args, stage="sft")
    
    # 创建训练器
    trainer = get_trainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        **kwargs
    )
    
    # 添加增强回调
    enhanced_callback = EnhancedTrainingCallback(monitor)
    trainer.add_callback(enhanced_callback)
    
    # 添加预测监控回调
    predict_callback = PredictMonitoringCallback(
        model_name=model_args.model_name_or_path,
        log_interval=training_args.logging_steps,
        save_predictions=True,
        detailed_analysis=True
    )
    trainer.add_callback(predict_callback)
    
    monitor.debugger.log_debug("✅ 增强训练器创建完成")
    monitor.debugger.log_debug(f"📊 数据集大小: {len(dataset) if dataset else 'N/A'}")
    monitor.debugger.log_debug(f"🔧 训练参数: {training_args}")
    
    return trainer, monitor

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强的LLaMA-Factory训练脚本")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型路径")
    parser.add_argument("--dataset_dir", type=str, default="data", help="数据集目录")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--cutoff_len", type=int, default=8192, help="截断长度")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=float, default=50.0, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--logging_steps", type=int, default=1, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=10, help="保存步数")
    parser.add_argument("--template", type=str, default="qwen3", help="模板")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="微调类型")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="all", help="LoRA target")
    parser.add_argument("--bf16", action="store_true", help="使用bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="梯度检查点")
    parser.add_argument("--flash_attn", type=str, default="auto", help="Flash attention")
    parser.add_argument("--trust_remote_code", action="store_true", help="信任远程代码")
    parser.add_argument("--overwrite_cache", action="store_true", help="覆盖缓存")
    parser.add_argument("--packing", action="store_true", help="打包")
    parser.add_argument("--enable_thinking", action="store_true", help="启用思考")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1, help="预处理工作进程数")
    parser.add_argument("--max_samples", type=int, default=100000, help="最大样本数")
    parser.add_argument("--warmup_steps", type=int, default=0, help="预热步数")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="优化器")
    parser.add_argument("--plot_loss", action="store_true", help="绘制损失")
    parser.add_argument("--include_num_input_tokens_seen", action="store_true", help="包含输入token数")
    parser.add_argument("--ddp_timeout", type=int, default=180000000, help="DDP超时")
    
    args = parser.parse_args()
    
    # 创建参数对象
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        flash_attn=args.flash_attn
    )
    
    data_args = DataArguments(
        dataset_dir=args.dataset_dir,
        dataset=args.dataset,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        packing=args.packing,
        enable_thinking=args.enable_thinking,
        overwrite_cache=args.overwrite_cache
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        plot_loss=args.plot_loss,
        include_num_input_tokens_seen=args.include_num_input_tokens_seen,
        ddp_timeout=args.ddp_timeout,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        save_safetensors=True,
        report_to=None  # 禁用wandb等外部日志
    )
    
    finetuning_args = FinetuningArguments(
        finetuning_type=args.finetuning_type,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target
    )
    
    generating_args = GeneratingArguments()
    
    # 创建增强训练器
    trainer, monitor = create_enhanced_trainer(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args
    )
    
    # 开始训练
    monitor.debugger.log_debug("🚀 开始训练")
    monitor.debugger.log_debug(f"{'=' * 60}")
    
    try:
        trainer.train()
        monitor.debugger.log_debug("✅ 训练成功完成")
    except Exception as e:
        monitor.debugger.log_debug(f"❌ 训练失败: {str(e)}")
        raise
    
    # 保存最终模型
    trainer.save_model()
    monitor.debugger.log_debug("💾 模型已保存")

if __name__ == "__main__":
    main()
