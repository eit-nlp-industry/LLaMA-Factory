# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .converter import DatasetConverter
from .data_utils import Role


if TYPE_CHECKING:
    from .parser import DatasetAttr
    from ..hparams import DataArguments


@dataclass
class CustomContextDatasetConverter(DatasetConverter):
    """
    自定义数据转换器，实现三段式prompt格式：
    1. context: 动态上下文信息（包含时间）
    2. input: 原始input内容
    3. instruction: 原始instruction内容
    """
    
    def _convert_time_format(self, time_str: str) -> str:
        """
        将时间字符串转换为 YYYYMMDD 格式
        
        Args:
            time_str: 输入的时间字符串（如"7月16日"）
        
        Returns:
            YYYYMMDD格式的时间字符串（如"20250716"）
        """
        # 如果已经是8位数字格式，直接返回
        if re.match(r'^\d{8}$', time_str):
            return time_str
            
        # 处理 "7月16日" 格式
        month_day_match = re.search(r'(\d{1,2})月(\d{1,2})日', time_str)
        if month_day_match:
            month = int(month_day_match.group(1))
            day = int(month_day_match.group(2))
            return f"2025{month:02d}{day:02d}"
            
        # 处理 "7.16" 格式
        dot_format_match = re.search(r'(\d{1,2})\.(\d{1,2})', time_str)
        if dot_format_match:
            month = int(dot_format_match.group(1))
            day = int(dot_format_match.group(2))
            return f"2025{month:02d}{day:02d}"
            
        # 处理 "2025-07-16" 格式
        iso_format_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', time_str)
        if iso_format_match:
            year = iso_format_match.group(1)
            month = int(iso_format_match.group(2))
            day = int(iso_format_match.group(3))
            return f"{year}{month:02d}{day:02d}"
            
        # 如果无法解析，返回默认值
        return "20250101"
    
    def _extract_context_from_data(self, example: dict) -> dict:
        """
        从训练数据中提取context信息
        
        Args:
            example: 训练样本
        
        Returns:
            context对象字典
        """
        context = {}
        
        # 优先使用新格式的context字段
        if 'context' in example and isinstance(example['context'], dict):
            context = example['context'].copy()
        # 兼容旧格式的context_time字段
        elif 'context_time' in example:
            time_formatted = self._convert_time_format(example['context_time'])
            context['time'] = time_formatted
        else:
            # 从input和instruction中提取时间信息（保留原有逻辑作为后备）
            input_text = example.get('input', '')
            instruction = example.get('instruction', '')
            time_str = self._extract_time_from_legacy_data(input_text, instruction)
            time_formatted = self._convert_time_format(time_str)
            context['time'] = time_formatted
            
        return context
    
    def _extract_time_from_legacy_data(self, input_text: str, instruction: str) -> str:
        """
        从输入文本和指令中提取时间信息（保留原有逻辑）
        
        Args:
            input_text: 当前样本的input
            instruction: 当前样本的instruction
        
        Returns:
            提取到的时间字符串
        """
        time_str = "未知时间"
        
        # 直接从input中提取日期信息
        date_patterns = [
            r'(\d{1,2}月\d{1,2}日)',  # 7月15日
            r'(\d{4}-\d{1,2}-\d{1,2})',  # 2025-07-15
            r'(\d{1,2}\.\d{1,2})',  # 7.15
            r'日期.*?(\d{1,2}\.\d{1,2})',  # 日期：7.16
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, input_text)
            if match:
                time_str = match.group(1)
                # 如果是数字格式，转换为更友好的格式
                if re.match(r'\d{1,2}\.\d{1,2}', time_str):
                    month, day = time_str.split('.')
                    time_str = f"{month}月{day}日"
                break
        
        return time_str
    
    def _build_context(self, context_dict: dict) -> str:
        """
        构建context部分
        
        Args:
            context_dict: context对象字典
            
        Returns:
            格式化的context字符串
        """
        context_json = json.dumps(context_dict, ensure_ascii=False)
        return f"context: {context_json}"
    
    def _format_three_part_prompt(self, instruction: str, input_text: str, context: str) -> str:
        """
        构建三段式prompt
        
        Args:
            instruction: 原始instruction
            input_text: 原始input
            context: 构建的context
            
        Returns:
            三段式格式的prompt
        """
        return f"{context}\n\ninput: {input_text}\n\ninstruction: {instruction}"
    
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        # 获取原始数据
        original_instruction = example.get(self.dataset_attr.prompt, "")  # instruction字段
        original_input = example.get(self.dataset_attr.query, "")         # input字段
        original_output = example.get(self.dataset_attr.response, "")     # output字段
        
        # 提取context信息
        context_dict = self._extract_context_from_data(example)
        
        # 构建context字符串
        context = self._build_context(context_dict)
        
        # 构建三段式prompt
        formatted_prompt = self._format_three_part_prompt(
            original_instruction,
            original_input, 
            context
        )
        
        # 构建prompt和response
        prompt = [{"role": Role.USER.value, "content": formatted_prompt}]
        
        if original_output:
            response = [{"role": Role.ASSISTANT.value, "content": original_output}]
        else:
            response = []
        
        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output
