#!/usr/bin/env python3
"""
训练过程中的标签和预测监控回调
直接集成到LLaMA-Factory的训练流程中
"""

import os
import json
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PredictionOutput

class LabelPredictionMonitor(TrainerCallback):
    """训练过程中的标签和预测监控回调"""
    
    def __init__(self, 
                 output_dir: str,
                 log_interval: int = 10,
                 save_detailed_logs: bool = True):
        """
        初始化监控回调
        
        Args:
            output_dir: 输出目录
            log_interval: 日志记录间隔（每N步记录一次）
            save_detailed_logs: 是否保存详细日志
        """
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_detailed_logs = save_detailed_logs
        
        # 创建日志目录
        self.log_dir = os.path.join(output_dir, "monitoring_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_files = {
            "labels": os.path.join(self.log_dir, f"label_analysis_{timestamp}.log"),
            "predictions": os.path.join(self.log_dir, f"prediction_monitor_{timestamp}.log"),
            "alignment": os.path.join(self.log_dir, f"alignment_analysis_{timestamp}.log"),
            "summary": os.path.join(self.log_dir, f"training_summary_{timestamp}.json")
        }
        
        # 设置日志记录器
        self.loggers = self._setup_loggers()
        
        # 存储历史数据
        self.training_history = []
        self.prediction_history = []
        self.step_count = 0
        
        # 记录初始化
        self.loggers["labels"].info("🔧 标签预测监控回调初始化完成")
        self.loggers["labels"].info(f"📁 输出目录: {output_dir}")
        self.loggers["labels"].info(f"📝 日志文件: {self.log_files['labels']}")
        self.loggers["labels"].info(f"📊 记录间隔: {log_interval}步")
    
    def _setup_loggers(self) -> Dict[str, logging.Logger]:
        """设置日志记录器"""
        loggers = {}
        
        for log_type, log_file in self.log_files.items():
            if log_type == "summary":
                continue  # summary是JSON文件，不需要logger
                
            logger = logging.getLogger(f"monitor_{log_type}")
            logger.setLevel(logging.INFO)
            
            # 清除现有处理器
            logger.handlers.clear()
            
            # 文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            loggers[log_type] = logger
        
        return loggers
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在每个训练步骤结束时调用"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self.loggers["labels"].info(f"\n{'='*80}")
            self.loggers["labels"].info(f"🔄 训练步骤 {state.global_step} 监控")
            self.loggers["labels"].info(f"{'='*80}")
            
            # 记录训练状态
            if state.log_history:
                latest_log = state.log_history[-1]
                self.loggers["labels"].info(f"📈 当前Loss: {latest_log.get('loss', 'N/A')}")
                self.loggers["labels"].info(f"📊 学习率: {latest_log.get('learning_rate', 'N/A')}")
                self.loggers["labels"].info(f"⏱️ 训练时间: {state.training_time:.2f}秒")
            
            # 记录步骤信息
            step_info = {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                "loss": state.log_history[-1].get('loss') if state.log_history else None,
                "learning_rate": state.log_history[-1].get('learning_rate') if state.log_history else None,
                "training_time": state.training_time
            }
            
            self.training_history.append(step_info)
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在评估时调用"""
        self.loggers["predictions"].info(f"\n{'='*80}")
        self.loggers["predictions"].info(f"📊 评估阶段监控 - 步骤 {state.global_step}")
        self.loggers["predictions"].info(f"{'='*80}")
        
        # 如果有预测结果，进行分析
        if hasattr(kwargs, 'predict_results') and kwargs['predict_results'] is not None:
            self._analyze_predictions(kwargs['predict_results'], state.global_step)
    
    def on_predict(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """在预测时调用"""
        self.loggers["predictions"].info(f"\n{'='*80}")
        self.loggers["predictions"].info(f"🔮 预测阶段监控 - 步骤 {state.global_step}")
        self.loggers["predictions"].info(f"{'='*80}")
        
        # 获取预测结果
        predict_results = kwargs.get('predict_results')
        if predict_results is not None:
            self._analyze_predictions(predict_results, state.global_step)
    
    def _analyze_predictions(self, predict_results: PredictionOutput, step: int):
        """分析预测结果"""
        self.loggers["predictions"].info(f"📊 预测结果分析 - 步骤 {step}")
        
        # 获取预测和标签
        predictions = predict_results.predictions
        labels = predict_results.label_ids
        
        if predictions is None or labels is None:
            self.loggers["predictions"].warning("⚠️ 预测结果或标签为空")
            return
        
        # 转换为numpy数组（如果是tensor）
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 分析每个样本
        batch_size = len(predictions)
        self.loggers["predictions"].info(f"📦 批次大小: {batch_size}")
        
        for i in range(min(batch_size, 3)):  # 只分析前3个样本
            self.loggers["predictions"].info(f"\n🔍 样本 {i+1} 分析:")
            
            pred_sample = predictions[i]
            label_sample = labels[i]
            
            # 移除padding
            pred_sample = self._remove_padding(pred_sample)
            label_sample = self._remove_padding(label_sample)
            
            # 记录标签信息
            self.loggers["labels"].info(f"\n📝 样本 {i+1} 标签分析:")
            self.loggers["labels"].info(f"  🎯 标签长度: {len(label_sample)}")
            self.loggers["labels"].info(f"  🔮 预测长度: {len(pred_sample)}")
            
            # 对齐分析
            alignment_analysis = self._analyze_alignment(pred_sample, label_sample)
            self.loggers["alignment"].info(f"\n🎯 样本 {i+1} 对齐分析:")
            self.loggers["alignment"].info(f"  📏 长度差异: {alignment_analysis['length_difference']}")
            self.loggers["alignment"].info(f"  🎯 精确匹配: {alignment_analysis['exact_match_percentage']:.1f}%")
            self.loggers["alignment"].info(f"  ✅ 有效匹配: {alignment_analysis['valid_match_percentage']:.1f}%")
            
            # 存储分析结果
            analysis = {
                "step": step,
                "sample_idx": i,
                "timestamp": datetime.now().isoformat(),
                "predictions": pred_sample.tolist(),
                "labels": label_sample.tolist(),
                "alignment_analysis": alignment_analysis
            }
            
            self.prediction_history.append(analysis)
    
    def _remove_padding(self, tokens: np.ndarray, pad_token_id: int = -100) -> np.ndarray:
        """移除padding tokens"""
        # 找到非padding的位置
        non_pad_mask = tokens != pad_token_id
        if np.any(non_pad_mask):
            # 找到第一个和最后一个非padding位置
            first_non_pad = np.argmax(non_pad_mask)
            last_non_pad = len(tokens) - 1 - np.argmax(non_pad_mask[::-1])
            return tokens[first_non_pad:last_non_pad+1]
        else:
            return tokens
    
    def _analyze_alignment(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """分析预测和标签的对齐情况"""
        # 基本统计
        min_len = min(len(predictions), len(labels))
        max_len = max(len(predictions), len(labels))
        
        # 计算匹配
        exact_matches = 0
        valid_matches = 0
        
        for i in range(min_len):
            if predictions[i] == labels[i]:
                exact_matches += 1
            if predictions[i] != -100 and labels[i] != -100:
                valid_matches += 1
        
        # 计算匹配率
        exact_match_percentage = (exact_matches / min_len * 100) if min_len > 0 else 0
        valid_match_percentage = (exact_matches / valid_matches * 100) if valid_matches > 0 else 0
        
        return {
            "min_length": min_len,
            "max_length": max_len,
            "exact_matches": exact_matches,
            "valid_matches": valid_matches,
            "exact_match_percentage": exact_match_percentage,
            "valid_match_percentage": valid_match_percentage,
            "length_difference": abs(len(predictions) - len(labels))
        }
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """训练结束时调用"""
        self.loggers["labels"].info(f"\n{'='*80}")
        self.loggers["labels"].info(f"🏁 训练结束监控")
        self.loggers["labels"].info(f"{'='*80}")
        
        # 保存最终分析摘要
        summary_data = {
            "training_info": {
                "total_steps": len(self.training_history),
                "total_predictions": len(self.prediction_history),
                "completion_time": datetime.now().isoformat(),
                "output_dir": self.output_dir
            },
            "training_history": self.training_history,
            "prediction_history": self.prediction_history,
            "log_files": self.log_files
        }
        
        with open(self.log_files["summary"], "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        self.loggers["labels"].info(f"📊 训练摘要已保存: {self.log_files['summary']}")
        self.loggers["labels"].info(f"📝 标签分析日志: {self.log_files['labels']}")
        self.loggers["labels"].info(f"🔮 预测监控日志: {self.log_files['predictions']}")
        self.loggers["labels"].info(f"🎯 对齐分析日志: {self.log_files['alignment']}")

def create_label_prediction_monitor(output_dir: str, **kwargs) -> LabelPredictionMonitor:
    """创建标签预测监控回调的工厂函数"""
    return LabelPredictionMonitor(output_dir=output_dir, **kwargs)
