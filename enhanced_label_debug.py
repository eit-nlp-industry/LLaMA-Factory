#!/usr/bin/env python3
"""
增强的标签调试器
用于训练过程中详细记录标签、预测变化和对齐情况
"""

import json
import os
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import AutoTokenizer
import logging

class EnhancedLabelDebugger:
    """增强的标签调试器"""
    
    def __init__(self, 
                 model_name: str,
                 log_file: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        初始化调试器
        
        Args:
            model_name: 模型名称或路径
            log_file: 日志文件路径
            log_level: 日志级别
        """
        self.model_name = model_name
        self.log_file = log_file or f"/home/ziqiang/LLaMA-Factory/enhanced_label_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 初始化tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"⚠️ 无法加载tokenizer: {e}")
            self.tokenizer = None
        
        # 设置日志
        self.logger = self._setup_logger(log_level)
        
        # 存储历史数据
        self.training_history = []
        self.prediction_history = []
        
        self.log_debug(f"🔧 增强标签调试器初始化完成")
        self.log_debug(f"📁 模型路径: {model_name}")
        self.log_debug(f"📄 日志文件: {self.log_file}")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"EnhancedLabelDebugger_{id(self)}")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # 避免重复添加handler
        if not logger.handlers:
            # 文件handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 控制台handler
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
        
        return logger
    
    def log_debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def log_info(self, message: str):
        """记录信息"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """记录错误"""
        self.logger.error(message)
    
    def analyze_training_step(self, 
                            step: int,
                            predictions: List[int],
                            labels: List[int],
                            loss: Optional[float] = None,
                            input_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        分析训练步骤
        
        Args:
            step: 训练步骤
            predictions: 预测token ids
            labels: 标签token ids
            loss: 损失值
            input_ids: 输入token ids
            
        Returns:
            分析结果字典
        """
        self.log_debug(f"\n{'='*80}")
        self.log_debug(f"🔍 训练步骤 {step} 详细分析")
        self.log_debug(f"{'='*80}")
        
        # 基本统计
        analysis = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": loss,
            "predictions_length": len(predictions),
            "labels_length": len(labels),
            "input_length": len(input_ids) if input_ids else 0
        }
        
        # 移除padding和特殊token
        clean_predictions = self._clean_tokens(predictions)
        clean_labels = self._clean_tokens(labels)
        clean_input = self._clean_tokens(input_ids) if input_ids else []
        
        analysis.update({
            "clean_predictions_length": len(clean_predictions),
            "clean_labels_length": len(clean_labels),
            "clean_input_length": len(clean_input)
        })
        
        # 解码文本
        if self.tokenizer:
            try:
                pred_text = self.tokenizer.decode(clean_predictions, skip_special_tokens=True)
                label_text = self.tokenizer.decode(clean_labels, skip_special_tokens=True)
                input_text = self.tokenizer.decode(clean_input, skip_special_tokens=True) if clean_input else ""
                
                analysis.update({
                    "prediction_text": pred_text,
                    "label_text": label_text,
                    "input_text": input_text
                })
                
                self.log_debug(f"📝 输入文本: {input_text[:200]}..." if len(input_text) > 200 else f"📝 输入文本: {input_text}")
                self.log_debug(f"🎯 标签文本: {label_text[:200]}..." if len(label_text) > 200 else f"🎯 标签文本: {label_text}")
                self.log_debug(f"🔮 预测文本: {pred_text[:200]}..." if len(pred_text) > 200 else f"🔮 预测文本: {pred_text}")
                
            except Exception as e:
                self.log_warning(f"⚠️ 解码失败: {e}")
                analysis.update({
                    "prediction_text": f"解码失败: {e}",
                    "label_text": f"解码失败: {e}",
                    "input_text": f"解码失败: {e}" if clean_input else ""
                })
        
        # 对齐分析
        alignment_analysis = self._analyze_alignment(clean_predictions, clean_labels)
        analysis["alignment_analysis"] = alignment_analysis
        
        # Token级别分析
        token_analysis = self._analyze_tokens(clean_predictions, clean_labels)
        analysis["token_analysis"] = token_analysis
        
        # 保存到历史
        self.training_history.append(analysis)
        
        # 打印摘要
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _clean_tokens(self, tokens: List[int]) -> List[int]:
        """清理tokens，移除padding和特殊token"""
        if not tokens:
            return []
        
        # 移除padding token
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else -100
        clean_tokens = [token for token in tokens if token != pad_token_id and token != -100]
        
        # 移除开头的特殊token
        if self.tokenizer:
            special_tokens = [
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.unk_token_id
            ]
            special_tokens = [tid for tid in special_tokens if tid is not None]
            
            while clean_tokens and clean_tokens[0] in special_tokens:
                clean_tokens.pop(0)
        
        return clean_tokens
    
    def _analyze_alignment(self, predictions: List[int], labels: List[int]) -> Dict[str, Any]:
        """分析预测和标签的对齐情况"""
        self.log_debug(f"\n📊 对齐分析:")
        
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
        
        alignment_analysis = {
            "min_length": min_len,
            "max_length": max_len,
            "exact_matches": exact_matches,
            "valid_matches": valid_matches,
            "exact_match_percentage": exact_match_percentage,
            "valid_match_percentage": valid_match_percentage,
            "length_difference": abs(len(predictions) - len(labels))
        }
        
        self.log_debug(f"  📏 长度: 预测={len(predictions)}, 标签={len(labels)}")
        self.log_debug(f"  🎯 精确匹配: {exact_matches}/{min_len} ({exact_match_percentage:.1f}%)")
        self.log_debug(f"  ✅ 有效匹配: {exact_matches}/{valid_matches} ({valid_match_percentage:.1f}%)")
        
        return alignment_analysis
    
    def _analyze_tokens(self, predictions: List[int], labels: List[int]) -> Dict[str, Any]:
        """分析token级别的差异"""
        self.log_debug(f"\n🔍 Token级别分析:")
        
        # 找到差异位置
        differences = []
        min_len = min(len(predictions), len(labels))
        
        for i in range(min_len):
            if predictions[i] != labels[i]:
                differences.append({
                    "position": i,
                    "prediction_token": predictions[i],
                    "label_token": labels[i],
                    "prediction_text": self._decode_single_token(predictions[i]),
                    "label_text": self._decode_single_token(labels[i])
                })
        
        # 统计token频率
        pred_token_counts = {}
        label_token_counts = {}
        
        for token in predictions:
            pred_token_counts[token] = pred_token_counts.get(token, 0) + 1
        
        for token in labels:
            label_token_counts[token] = label_token_counts.get(token, 0) + 1
        
        token_analysis = {
            "differences": differences[:10],  # 只保存前10个差异
            "total_differences": len(differences),
            "prediction_token_counts": dict(list(pred_token_counts.items())[:20]),  # 前20个
            "label_token_counts": dict(list(label_token_counts.items())[:20])
        }
        
        self.log_debug(f"  🔍 差异位置数: {len(differences)}")
        if differences:
            self.log_debug(f"  📍 前5个差异:")
            for i, diff in enumerate(differences[:5]):
                self.log_debug(f"    位置{diff['position']}: 预测='{diff['prediction_text']}' vs 标签='{diff['label_text']}'")
        
        return token_analysis
    
    def _decode_single_token(self, token_id: int) -> str:
        """解码单个token"""
        if not self.tokenizer or token_id == -100:
            return f"<{token_id}>"
        
        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=True)
        except:
            return f"<{token_id}>"
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """打印分析摘要"""
        self.log_debug(f"\n📋 分析摘要:")
        self.log_debug(f"  📊 步骤: {analysis['step']}")
        self.log_debug(f"  📉 损失: {analysis['loss']:.6f}" if analysis['loss'] else "  📉 损失: N/A")
        self.log_debug(f"  📏 长度: 预测={analysis['clean_predictions_length']}, 标签={analysis['clean_labels_length']}")
        
        alignment = analysis['alignment_analysis']
        self.log_debug(f"  🎯 匹配率: {alignment['exact_match_percentage']:.1f}%")
        self.log_debug(f"  ✅ 有效匹配: {alignment['valid_match_percentage']:.1f}%")
        
        token_analysis = analysis['token_analysis']
        self.log_debug(f"  🔍 差异数: {token_analysis['total_differences']}")
    
    def analyze_prediction_changes(self, 
                                 step: int,
                                 current_predictions: List[int],
                                 previous_predictions: Optional[List[int]] = None) -> Dict[str, Any]:
        """分析预测变化"""
        if previous_predictions is None:
            return {"step": step, "changes": "无历史数据"}
        
        self.log_debug(f"\n🔄 预测变化分析 - 步骤 {step}")
        self.log_debug(f"{'='*60}")
        
        # 计算变化
        changes = []
        min_len = min(len(current_predictions), len(previous_predictions))
        
        for i in range(min_len):
            if current_predictions[i] != previous_predictions[i]:
                changes.append({
                    "position": i,
                    "old_token": previous_predictions[i],
                    "new_token": current_predictions[i],
                    "old_text": self._decode_single_token(previous_predictions[i]),
                    "new_text": self._decode_single_token(current_predictions[i])
                })
        
        # 长度变化
        length_change = len(current_predictions) - len(previous_predictions)
        
        change_analysis = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "changes": changes[:10],  # 只保存前10个变化
            "total_changes": len(changes),
            "length_change": length_change,
            "change_percentage": (len(changes) / min_len * 100) if min_len > 0 else 0
        }
        
        self.log_debug(f"📊 变化统计:")
        self.log_debug(f"  🔄 总变化数: {len(changes)}")
        self.log_debug(f"  📏 长度变化: {length_change:+d}")
        self.log_debug(f"  📈 变化率: {change_analysis['change_percentage']:.1f}%")
        
        if changes:
            self.log_debug(f"  📍 前5个变化:")
            for i, change in enumerate(changes[:5]):
                self.log_debug(f"    位置{change['position']}: '{change['old_text']}' -> '{change['new_text']}'")
        
        # 保存到历史
        self.prediction_history.append(change_analysis)
        
        return change_analysis
    
    def save_analysis_summary(self, analyses: List[Dict[str, Any]], output_file: str):
        """保存分析摘要"""
        summary = {
            "model_name": self.model_name,
            "analysis_time": datetime.now().isoformat(),
            "total_steps": len(analyses),
            "analyses": analyses,
            "summary_statistics": self._calculate_summary_statistics(analyses)
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.log_info(f"📊 分析摘要已保存到: {output_file}")
    
    def _calculate_summary_statistics(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算摘要统计"""
        if not analyses:
            return {}
        
        losses = [a["loss"] for a in analyses if a["loss"] is not None]
        match_percentages = [a["alignment_analysis"]["exact_match_percentage"] for a in analyses]
        valid_match_percentages = [a["alignment_analysis"]["valid_match_percentage"] for a in analyses]
        
        return {
            "loss_statistics": {
                "min": min(losses) if losses else None,
                "max": max(losses) if losses else None,
                "mean": sum(losses) / len(losses) if losses else None,
                "final": losses[-1] if losses else None
            },
            "match_statistics": {
                "exact_match_percentage": {
                    "min": min(match_percentages) if match_percentages else None,
                    "max": max(match_percentages) if match_percentages else None,
                    "mean": sum(match_percentages) / len(match_percentages) if match_percentages else None,
                    "final": match_percentages[-1] if match_percentages else None
                },
                "valid_match_percentage": {
                    "min": min(valid_match_percentages) if valid_match_percentages else None,
                    "max": max(valid_match_percentages) if valid_match_percentages else None,
                    "mean": sum(valid_match_percentages) / len(valid_match_percentages) if valid_match_percentages else None,
                    "final": valid_match_percentages[-1] if valid_match_percentages else None
                }
            }
        }
    
    def generate_training_report(self, output_file: str):
        """生成训练报告"""
        report = {
            "model_name": self.model_name,
            "report_time": datetime.now().isoformat(),
            "training_history": self.training_history,
            "prediction_history": self.prediction_history,
            "summary": self._calculate_summary_statistics(self.training_history)
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.log_info(f"📋 训练报告已生成: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 创建调试器
    debugger = EnhancedLabelDebugger(
        model_name="/data/models/Qwen3-8B",
        log_file="/home/ziqiang/LLaMA-Factory/test_debug.log"
    )
    
    # 模拟分析
    sample_predictions = [1, 2, 3, 4, 5]
    sample_labels = [1, 2, 6, 4, 5]
    sample_input = [0, 1, 2, 3, 4, 5]
    
    analysis = debugger.analyze_training_step(
        step=1,
        predictions=sample_predictions,
        labels=sample_labels,
        loss=0.5,
        input_ids=sample_input
    )
    
    print("✅ 调试器测试完成")
    print(f"📁 日志文件: {debugger.log_file}")
