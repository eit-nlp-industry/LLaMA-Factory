#!/usr/bin/env python3
"""
训练过程中的Predict监控回调
用于实时监控训练过程中predict的变化和对齐情况
"""

import json
import os
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PredictionOutput

from enhanced_label_debug import EnhancedLabelDebugger

class PredictMonitoringCallback(TrainerCallback):
    """训练过程中的Predict监控回调"""
    
    def __init__(self, 
                 model_name: str,
                 log_interval: int = 10,
                 save_predictions: bool = True,
                 detailed_analysis: bool = True):
        """
        初始化监控回调
        
        Args:
            model_name: 模型名称
            log_interval: 日志记录间隔（每N步记录一次）
            save_predictions: 是否保存预测结果
            detailed_analysis: 是否进行详细分析
        """
        self.model_name = model_name
        self.log_interval = log_interval
        self.save_predictions = save_predictions
        self.detailed_analysis = detailed_analysis
        
        # 初始化调试器
        self.debugger = EnhancedLabelDebugger(
            model_name=model_name,
            log_file=f"/home/ziqiang/LLaMA-Factory/training_predict_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # 存储历史数据
        self.step_analyses = []
        self.prediction_history = []
        
        self.debugger.log_debug(f"🔧 Predict监控回调初始化完成")
        self.debugger.log_debug(f"📊 日志间隔: {log_interval}步")
        self.debugger.log_debug(f"💾 保存预测: {save_predictions}")
        self.debugger.log_debug(f"🔍 详细分析: {detailed_analysis}")
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在每个训练步骤结束时调用"""
        if state.global_step % self.log_interval == 0:
            self.debugger.log_debug(f"\n🔄 训练步骤 {state.global_step} 监控")
            self.debugger.log_debug(f"{'=' * 60}")
            
            # 记录训练状态
            self.debugger.log_debug(f"📈 当前Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")
            self.debugger.log_debug(f"📊 学习率: {state.log_history[-1].get('learning_rate', 'N/A') if state.log_history else 'N/A'}")
            self.debugger.log_debug(f"⏱️ 训练时间: {state.training_time:.2f}秒")
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在评估时调用"""
        self.debugger.log_debug(f"\n📊 评估阶段监控")
        self.debugger.log_debug(f"{'=' * 60}")
        self.debugger.log_debug(f"🔄 评估步骤: {state.global_step}")
        
        # 如果有预测结果，进行分析
        if hasattr(kwargs, 'predict_results') and kwargs['predict_results'] is not None:
            self._analyze_predictions(kwargs['predict_results'], state.global_step)
    
    def on_predict(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在预测时调用"""
        self.debugger.log_debug(f"\n🔮 预测阶段监控")
        self.debugger.log_debug(f"{'=' * 60}")
        
        # 获取预测结果
        predict_results = kwargs.get('predict_results')
        if predict_results is not None:
            self._analyze_predictions(predict_results, state.global_step)
    
    def _analyze_predictions(self, predict_results: PredictionOutput, step: int):
        """分析预测结果"""
        self.debugger.log_debug(f"📊 预测结果分析 - 步骤 {step}")
        
        # 获取预测和标签
        predictions = predict_results.predictions
        labels = predict_results.label_ids
        
        if predictions is None or labels is None:
            self.debugger.log_debug("⚠️ 预测结果或标签为空")
            return
        
        # 转换为numpy数组（如果是tensor）
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 分析每个样本
        batch_size = len(predictions)
        self.debugger.log_debug(f"📦 批次大小: {batch_size}")
        
        for i in range(min(batch_size, 3)):  # 只分析前3个样本
            self.debugger.log_debug(f"\n🔍 样本 {i+1} 分析:")
            
            pred_sample = predictions[i]
            label_sample = labels[i]
            
            # 移除padding
            pred_sample = self._remove_padding(pred_sample)
            label_sample = self._remove_padding(label_sample)
            
            # 进行详细分析
            if self.detailed_analysis:
                analysis = self.debugger.analyze_training_step(
                    step=step,
                    predictions=pred_sample.tolist(),
                    labels=label_sample.tolist(),
                    loss=predict_results.metrics.get('eval_loss', None) if hasattr(predict_results, 'metrics') else None
                )
                
                # 存储分析结果
                self.step_analyses.append(analysis)
                
                # 存储预测历史
                self.prediction_history.append({
                    "step": step,
                    "sample_idx": i,
                    "predictions": pred_sample.tolist(),
                    "labels": label_sample.tolist(),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 保存预测结果
        if self.save_predictions:
            self._save_predictions(predict_results, step)
    
    def _remove_padding(self, tokens: np.ndarray, pad_token_id: int = None) -> np.ndarray:
        """移除padding tokens"""
        if pad_token_id is None:
            pad_token_id = self.debugger.tokenizer.pad_token_id
        
        # 找到非padding的位置
        non_pad_mask = tokens != pad_token_id
        if np.any(non_pad_mask):
            # 找到第一个和最后一个非padding位置
            first_non_pad = np.argmax(non_pad_mask)
            last_non_pad = len(tokens) - 1 - np.argmax(non_pad_mask[::-1])
            return tokens[first_non_pad:last_non_pad+1]
        else:
            return tokens
    
    def _save_predictions(self, predict_results: PredictionOutput, step: int):
        """保存预测结果"""
        output_dir = "/home/ziqiang/LLaMA-Factory/prediction_monitoring"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始预测结果
        pred_file = os.path.join(output_dir, f"predictions_step_{step}.json")
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "predictions": predict_results.predictions.tolist() if isinstance(predict_results.predictions, np.ndarray) else predict_results.predictions,
                "label_ids": predict_results.label_ids.tolist() if isinstance(predict_results.label_ids, np.ndarray) else predict_results.label_ids,
                "metrics": predict_results.metrics if hasattr(predict_results, 'metrics') else {}
            }, f, ensure_ascii=False, indent=2)
        
        self.debugger.log_debug(f"💾 预测结果已保存到: {pred_file}")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时调用"""
        self.debugger.log_debug(f"\n🏁 训练结束监控")
        self.debugger.log_debug(f"{'=' * 60}")
        
        # 保存最终分析摘要
        if self.step_analyses:
            self.debugger.save_analysis_summary(
                self.step_analyses,
                f"/home/ziqiang/LLaMA-Factory/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        # 生成训练趋势分析
        self._generate_training_trends()
    
    def _generate_training_trends(self):
        """生成训练趋势分析"""
        if not self.step_analyses:
            return
        
        self.debugger.log_debug(f"\n📈 训练趋势分析")
        self.debugger.log_debug(f"{'=' * 60}")
        
        # 提取关键指标
        steps = [analysis["step"] for analysis in self.step_analyses]
        losses = [analysis["loss"] for analysis in self.step_analyses if analysis["loss"] is not None]
        valid_match_percentages = [
            analysis["alignment_analysis"]["valid_match_percentage"] 
            for analysis in self.step_analyses
        ]
        
        if losses:
            self.debugger.log_debug(f"📉 Loss趋势: {min(losses):.6f} -> {max(losses):.6f}")
        
        if valid_match_percentages:
            self.debugger.log_debug(f"🎯 有效匹配率趋势: {min(valid_match_percentages):.1f}% -> {max(valid_match_percentages):.1f}%")
        
        # 保存趋势数据
        trend_data = {
            "steps": steps,
            "losses": losses,
            "valid_match_percentages": valid_match_percentages,
            "analysis_time": datetime.now().isoformat()
        }
        
        trend_file = f"/home/ziqiang/LLaMA-Factory/training_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trend_file, "w", encoding="utf-8") as f:
            json.dump(trend_data, f, ensure_ascii=False, indent=2)
        
        self.debugger.log_debug(f"📊 趋势数据已保存到: {trend_file}")

def create_predict_monitoring_callback(model_name: str, **kwargs) -> PredictMonitoringCallback:
    """创建预测监控回调的工厂函数"""
    return PredictMonitoringCallback(model_name=model_name, **kwargs)

# 使用示例
if __name__ == "__main__":
    # 创建监控回调
    callback = create_predict_monitoring_callback(
        model_name="/data/models/Qwen3-8B",
        log_interval=5,
        save_predictions=True,
        detailed_analysis=True
    )
    
    print("✅ Predict监控回调创建完成")
    print(f"📁 日志文件: {callback.debugger.log_file}")
    print(f"📊 监控间隔: {callback.log_interval}步")
