#!/usr/bin/env python3
"""
自定义数据处理器 - 支持动态mask user_id字段
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ...extras.constants import IGNORE_INDEX
from .supervised import SupervisedDataProcessor


class UserIdMaskDataProcessor(SupervisedDataProcessor):
    """支持动态mask user_id字段的数据处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_id_patterns = [
            r'"user_id"\s*:\s*\d+',  # "user_id": 136451106
            r'"user_id"\s*:\s*"(\d+)"',  # "user_id": "136451106"
        ]
    
    def _mask_user_id_in_text(self, text: str) -> Tuple[str, List[int]]:
        """
        在文本中mask掉user_id字段，返回mask后的文本和mask位置
        
        Args:
            text: 原始文本
            
        Returns:
            Tuple[str, List[int]]: (mask后的文本, mask的token位置列表)
        """
        masked_text = text
        mask_positions = []
        
        for pattern in self.user_id_patterns:
            matches = list(re.finditer(pattern, masked_text))
            for match in reversed(matches):  # 从后往前替换，避免位置偏移
                start, end = match.span()
                # 替换为占位符
                masked_text = masked_text[:start] + '"user_id": <MASKED>' + masked_text[end:]
        
        return masked_text, mask_positions
    
    def _find_user_id_token_positions(self, input_ids: List[int], tokenizer) -> List[int]:
        """
        找到input_ids中user_id对应的token位置
        
        Args:
            input_ids: 输入的token ID列表
            tokenizer: tokenizer对象
            
        Returns:
            List[int]: user_id对应的token位置列表
        """
        # 将input_ids解码为文本
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # 找到user_id的位置
        user_id_positions = []
        for pattern in self.user_id_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                start_char, end_char = match.span()
                
                # 将字符位置转换为token位置
                # 这是一个简化的实现，实际可能需要更复杂的映射
                start_token = len(tokenizer.encode(text[:start_char], add_special_tokens=False))
                end_token = len(tokenizer.encode(text[:end_char], add_special_tokens=False))
                
                user_id_positions.extend(range(start_token, end_token))
        
        return user_id_positions
    
    def _apply_user_id_mask(self, input_ids: List[int], labels: List[int], tokenizer) -> List[int]:
        """
        在labels中mask掉user_id对应的token位置
        
        Args:
            input_ids: 输入的token ID列表
            labels: 标签列表
            tokenizer: tokenizer对象
            
        Returns:
            List[int]: mask后的labels
        """
        masked_labels = labels.copy()
        
        # 找到user_id的token位置
        user_id_positions = self._find_user_id_token_positions(input_ids, tokenizer)
        
        # 将user_id位置的labels设为IGNORE_INDEX
        for pos in user_id_positions:
            if 0 <= pos < len(masked_labels):
                masked_labels[pos] = IGNORE_INDEX
        
        return masked_labels
    
    def process_on_prompt(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """处理单轮对话数据，应用user_id mask"""
        # 调用父类方法获取基本处理结果
        result = super().process_on_prompt(example)
        
        # 应用user_id mask
        if "input_ids" in result and "labels" in result:
            input_ids = result["input_ids"]
            labels = result["labels"]
            
            # 应用user_id mask
            masked_labels = self._apply_user_id_mask(input_ids, labels, self.tokenizer)
            result["labels"] = masked_labels
            
            # 记录mask信息
            original_trainable = sum(1 for label in labels if label != IGNORE_INDEX)
            masked_trainable = sum(1 for label in masked_labels if label != IGNORE_INDEX)
            masked_count = original_trainable - masked_trainable
            
            if masked_count > 0:
                print(f"🔒 已mask {masked_count} 个user_id相关token")
                print(f"   原始可训练token: {original_trainable}")
                print(f"   mask后可训练token: {masked_trainable}")
        
        return result
    
    def process_on_pack(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理打包数据，应用user_id mask"""
        # 调用父类方法获取基本处理结果
        result = super().process_on_pack(examples)
        
        # 应用user_id mask
        if "input_ids" in result and "labels" in result:
            input_ids = result["input_ids"]
            labels = result["labels"]
            
            # 应用user_id mask
            masked_labels = self._apply_user_id_mask(input_ids, labels, self.tokenizer)
            result["labels"] = masked_labels
            
            # 记录mask信息
            original_trainable = sum(1 for label in labels if label != IGNORE_INDEX)
            masked_trainable = sum(1 for label in masked_labels if label != IGNORE_INDEX)
            masked_count = original_trainable - masked_trainable
            
            if masked_count > 0:
                print(f"🔒 已mask {masked_count} 个user_id相关token")
                print(f"   原始可训练token: {original_trainable}")
                print(f"   mask后可训练token: {masked_trainable}")
        
        return result
