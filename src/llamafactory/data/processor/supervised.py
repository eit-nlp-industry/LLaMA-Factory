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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
import re

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        
        # 添加详细日志记录
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
        
        log_debug("\n" + "🔧 " + "=" * 78)
        log_debug("🔧 ShareGPT数据处理器 - _encode_data_example开始")
        log_debug("🔧 " + "=" * 78)
        
        log_debug(f"📊 开始处理数据样本")
        log_debug(f"📊 原始conversations长度: {len(prompt + response)} 条消息")
        log_debug(f"📊 编码后的pairs数量: {len(encoded_pairs)}")
        log_debug(f"📊 初始total_length: {total_length}")
        log_debug(f"📊 cutoff_len: {self.data_args.cutoff_len}")
        log_debug(f"📊 mask_history: {self.data_args.mask_history}")
        log_debug(f"📊 train_on_prompt: {self.data_args.train_on_prompt}")
        
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns
            log_debug(f"🔄 启用mask_history，pairs顺序已反转")

        log_debug("\n" + "📋 " + "-" * 76)
        log_debug("📋 开始处理每个Pair")
        log_debug("📋 " + "-" * 76)

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            original_source_len = len(source_ids)
            original_target_len = len(target_ids)
            remaining_budget = self.data_args.cutoff_len - total_length
            
            log_debug(f"\n🔄 === 处理Pair {turn_idx + 1} ===")
            log_debug(f"📏 原始长度: source={original_source_len}, target={original_target_len}")
            log_debug(f"💰 剩余预算: {remaining_budget}")
            
            if total_length >= self.data_args.cutoff_len:
                log_debug(f"❌ 预算耗尽，丢弃剩余pairs")
                break

            source_len, target_len = infer_seqlen(
                original_source_len, original_target_len, remaining_budget
            )
            
            log_debug(f"✂️ 截断后长度: source={original_source_len}->{source_len}, target={original_target_len}->{target_len}")
            
            if source_len < original_source_len:
                log_debug(f"⚠️ source被截断: {original_source_len - source_len} tokens")
            if target_len < original_target_len:
                log_debug(f"⚠️ target被截断: {original_target_len - target_len} tokens")
            
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len
            
            log_debug(f"📈 当前累计长度: {total_length}/{self.data_args.cutoff_len} ({total_length/self.data_args.cutoff_len*100:.1f}%)")

            # 生成标签
            if self.data_args.train_on_prompt:
                source_label = source_ids
                log_debug(f"🏷️ train_on_prompt=True, source_label使用原始tokens")
                log_debug(f"   📊 source_label长度: {len(source_label)} tokens")
                if len(source_label) > 0:
                    source_label_preview = self.tokenizer.decode(source_label[:min(20, len(source_label))], skip_special_tokens=False)
                    source_label_clean = source_label_preview.replace(chr(10), '\\n')
                    log_debug(f"   📄 source_label预览: {source_label_clean[:100]}...")
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
                log_debug(f"🏷️ efficient_eos=True, source_label=[eos_token, {source_len-1}*IGNORE_INDEX]")
                log_debug(f"   📊 source_label长度: {len(source_label)} tokens")
                log_debug(f"   🔍 eos_token_id: {self.tokenizer.eos_token_id}")
                log_debug(f"   🔍 IGNORE_INDEX: {IGNORE_INDEX}")
            else:
                source_label = [IGNORE_INDEX] * source_len
                log_debug(f"🏷️ source_label={source_len}*IGNORE_INDEX")
                log_debug(f"   📊 source_label长度: {len(source_label)} tokens")
                log_debug(f"   🔍 IGNORE_INDEX: {IGNORE_INDEX}")

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
                log_debug(f"🏷️ mask_history=True且turn_idx!=0, target_label={target_len}*IGNORE_INDEX")
                log_debug(f"   📊 target_label长度: {len(target_label)} tokens")
                log_debug(f"   🔍 IGNORE_INDEX: {IGNORE_INDEX}")
            else:
                target_label = target_ids
                log_debug(f"🏷️ target_label使用原始tokens")
                log_debug(f"   📊 target_label长度: {len(target_label)} tokens")
                if len(target_label) > 0:
                    target_label_preview = self.tokenizer.decode(target_label[:min(20, len(target_label))], skip_special_tokens=False)
                    target_label_clean = target_label_preview.replace(chr(10), '\\n')
                    log_debug(f"   📄 target_label预览: {target_label_clean[:100]}...")

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
                log_debug(f"🔄 mask_history=True, 序列已反转拼接")
                log_debug(f"   📊 拼接后input_ids长度: {len(input_ids)}")
                log_debug(f"   📊 拼接后labels长度: {len(labels)}")
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
                log_debug(f"➡️ 正常顺序拼接")
                log_debug(f"   📊 拼接后input_ids长度: {len(input_ids)}")
                log_debug(f"   📊 拼接后labels长度: {len(labels)}")
            
            # 显示当前labels中的有效token统计
            valid_labels_count = sum(1 for label in labels if label != IGNORE_INDEX)
            total_labels_count = len(labels)
            valid_percentage = (valid_labels_count / total_labels_count * 100) if total_labels_count > 0 else 0
            log_debug(f"   📊 当前有效labels: {valid_labels_count}/{total_labels_count} ({valid_percentage:.1f}%)")
            
            # 显示labels的详细组成
            if len(labels) > 0:
                unique_labels = set(labels)
                label_stats = {}
                for label in unique_labels:
                    count = labels.count(label)
                    if label == IGNORE_INDEX:
                        label_stats[f"IGNORE_INDEX({label})"] = count
                    elif label == self.tokenizer.eos_token_id:
                        label_stats[f"EOS_TOKEN({label})"] = count
                    else:
                        label_stats[f"TOKEN_{label}"] = count
                
                log_debug(f"   📊 Labels组成: {dict(list(label_stats.items())[:5])}")  # 只显示前5个

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
            total_length += 1
            log_debug(f"🔚 添加eos_token, total_length={total_length}")

        log_debug("\n" + "🎯 " + "=" * 76)
        log_debug("🎯 最终结果统计")
        log_debug("🎯 " + "=" * 76)
        log_debug(f"📊 最终input_ids长度: {len(input_ids)}")
        log_debug(f"📊 最终labels长度: {len(labels)}")
        log_debug(f"📊 最终total_length: {total_length}")
        log_debug(f"📊 使用率: {total_length}/{self.data_args.cutoff_len} ({total_length/self.data_args.cutoff_len*100:.1f}%)")
        
        # 统计有效标签数量
        valid_labels = [l for l in labels if l != IGNORE_INDEX]
        log_debug(f"📊 有效标签数量: {len(valid_labels)}/{len(labels)} ({len(valid_labels)/len(labels)*100:.1f}%)")
        
        log_debug("🔧 " + "=" * 78)
        log_debug("🔧 _encode_data_example处理完成")
        log_debug("🔧 " + "=" * 78)

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        
        # 添加日志记录
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
        
        log_debug("\n" + "🚀 " + "=" * 78)
        log_debug("🚀 SupervisedDatasetProcessor.preprocess_dataset 开始")
        log_debug("🚀 " + "=" * 78)
        log_debug(f"📊 处理样本数量: {len(examples['_prompt'])}")
        
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            log_debug(f"\n🔄 处理样本 {i+1}/{len(examples['_prompt'])}")
            
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                log_debug(f"❌ 样本 {i+1} 格式无效，跳过")
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            log_debug(f"✅ 样本 {i+1} 格式有效，开始编码")
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            log_debug(f"✅ 样本 {i+1} 编码完成，input_ids长度: {len(input_ids)}, labels长度: {len(labels)}")
            
            # 应用user_id mask
            masked_labels = self._mask_user_id_tokens(input_ids, labels)
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(masked_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        log_debug("\n" + "🎯 " + "=" * 76)
        log_debug("🎯 preprocess_dataset 处理完成")
        log_debug("🎯 " + "=" * 76)
        log_debug(f"📊 最终处理样本数量: {len(model_inputs['input_ids'])}")
        log_debug("🚀 " + "=" * 78)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

    def _mask_user_id_tokens(self, input_ids: list[int], labels: list[int]) -> list[int]:
        """
        在labels中mask掉user_id对应的token位置
        
        Args:
            input_ids: 输入的token ID列表
            labels: 标签列表
            
        Returns:
            list[int]: mask后的labels
        """
        masked_labels = labels.copy()
        
        # 将input_ids解码为文本
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # 定义user_id的模式
        user_id_patterns = [
            r'"user_id"\s*:\s*\d+',  # "user_id": 136451106
            r'"user_id"\s*:\s*"(\d+)"',  # "user_id": "136451106"
        ]
        
        # 找到user_id的位置
        user_id_positions = []
        for pattern in user_id_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                start_char, end_char = match.span()
                
                # 使用更精确的方法找到token位置
                try:
                    # 获取user_id部分的文本
                    user_id_text = text[start_char:end_char]
                    
                    # 在input_ids中搜索这个文本对应的token序列
                    # 先尝试直接匹配
                    user_id_tokens = self.tokenizer.encode(user_id_text, add_special_tokens=False)
                    
                    # 在input_ids中查找这个token序列
                    for i in range(len(input_ids) - len(user_id_tokens) + 1):
                        if input_ids[i:i+len(user_id_tokens)] == user_id_tokens:
                            user_id_positions.extend(range(i, i+len(user_id_tokens)))
                            print(f"🔒 找到user_id token位置: {i} 到 {i+len(user_id_tokens)-1}")
                            print(f"   user_id文本: {user_id_text}")
                            print(f"   user_id tokens: {user_id_tokens}")
                            break
                    
                    # 如果直接匹配失败，尝试更宽松的匹配
                    if not user_id_positions:
                        # 提取数字部分
                        numbers = re.findall(r'\d+', user_id_text)
                        for num in numbers:
                            num_tokens = self.tokenizer.encode(num, add_special_tokens=False)
                            for i in range(len(input_ids) - len(num_tokens) + 1):
                                if input_ids[i:i+len(num_tokens)] == num_tokens:
                                    user_id_positions.extend(range(i, i+len(num_tokens)))
                                    print(f"🔒 找到数字token位置: {i} 到 {i+len(num_tokens)-1}")
                                    print(f"   数字: {num}")
                                    print(f"   数字tokens: {num_tokens}")
                                    break
                            if user_id_positions:
                                break
                                
                except Exception as e:
                    print(f"⚠️ user_id mask失败: {e}")
                    continue
        
        # 将user_id位置的labels设为IGNORE_INDEX
        for pos in user_id_positions:
            if 0 <= pos < len(masked_labels):
                masked_labels[pos] = IGNORE_INDEX
        
        # 记录mask信息
        original_trainable = sum(1 for label in labels if label != IGNORE_INDEX)
        masked_trainable = sum(1 for label in masked_labels if label != IGNORE_INDEX)
        masked_count = original_trainable - masked_trainable
        
        if masked_count > 0:
            print(f"🔒 已mask {masked_count} 个user_id相关token")
            print(f"   原始可训练token: {original_trainable}")
            print(f"   mask后可训练token: {masked_trainable}")
        else:
            print(f"⚠️ 未找到user_id token进行mask")
            print(f"   文本内容: {text[:200]}...")
        
        return masked_labels


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                # 应用user_id mask
                masked_labels = self._mask_user_id_tokens(input_ids, labels)
                
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(masked_labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
