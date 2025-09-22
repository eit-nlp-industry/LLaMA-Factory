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

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    import os
    from datetime import datetime
    
    def log_debug(msg):
        """简单的调试日志函数"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} | INFO | {msg}\n"
        
        # 写入日志文件
        log_file = "/home/ziqiang/LLaMA-Factory/sharegpt_pair_debug.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
                f.flush()  # 立即刷新到文件
        except:
            pass  # 忽略写文件错误
            
        # 只写入日志文件，不输出到控制台
    
    log_debug(f"🧮 输入: source_len={source_len}, target_len={target_len}, cutoff_len={cutoff_len}")
    
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
        strategy = "🎯 策略1: target完全保留，截断source"
        log_debug(f"📋 条件1: target_len*2 < cutoff_len ({target_len}*2={target_len*2} < {cutoff_len})")
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
        strategy = "🎯 策略2: source完全保留，截断target"
        log_debug(f"📋 条件2: source_len*2 < cutoff_len ({source_len}*2={source_len*2} < {cutoff_len})")
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
        strategy = "🎯 策略3: 按比例截断source和target"
        ratio = target_len / (source_len + target_len)
        log_debug(f"📋 条件3: 按比例截断，ratio={ratio:.3f}")

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    
    log_debug(f"{strategy}")
    log_debug(f"📊 输出: source_len={source_len}->{new_source_len}, target_len={target_len}->{new_target_len}")
    
    return new_source_len, new_target_len
