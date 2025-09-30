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
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key) for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

class LabelPredictionMonitorCallback(TrainerCallback):
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
        from datetime import datetime
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
        self.previous_tokens = {}
        self.previous_predictions = {}  # 存储之前的预测结果
        self.previous_prediction_texts = {}  # 存储之前的预测文本
        self.tokenizer = None  # 稍后从训练器获取
        
        # 记录初始化
        self.loggers["labels"].info("🔧 标签预测监控回调初始化完成")
        self.loggers["labels"].info(f"📁 输出目录: {output_dir}")
        self.loggers["labels"].info(f"📝 日志文件: {self.log_files['labels']}")
        self.loggers["labels"].info(f"📊 记录间隔: {log_interval}步")
    
    def _setup_loggers(self):
        """设置日志记录器"""
        import logging
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
    
    def set_tokenizer(self, tokenizer):
        """设置tokenizer用于Token解码"""
        self.tokenizer = tokenizer
        self.loggers["labels"].info(f"🔤 Tokenizer已设置: {type(tokenizer).__name__}")
    
    def _decode_tokens(self, token_ids):
        """解码Token ID为文本"""
        if self.tokenizer is None:
            return [f"<token_{tid}>" for tid in token_ids]
        
        try:
            # 解码Token
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            return decoded_text
        except Exception as e:
            return [f"<decode_error_{tid}>" for tid in token_ids]
    
    def analyze_model_predictions(self, model, inputs, labels):
        """分析模型的预测输出"""
        import torch
        import numpy as np
        
        # 只在特定步骤记录详细信息
        if self.step_count % self.log_interval == 0:
            self.loggers["labels"].info(f"\n🔮 模型预测分析 - 步骤 {self.step_count}")
            self.loggers["labels"].info(f"{'='*80}")
        
        try:
            # 获取input_ids
            if isinstance(inputs, dict):
                input_ids = inputs.get('input_ids')
            else:
                input_ids = inputs
            
            # 处理BatchEncoding对象
            if hasattr(input_ids, 'input_ids'):
                actual_input_ids = input_ids.input_ids
            else:
                actual_input_ids = input_ids
            
            if actual_input_ids is not None and labels is not None:
                # 设置模型为评估模式以获取预测
                model.eval()
                
                with torch.no_grad():
                    # 获取模型输出
                    outputs = model(actual_input_ids)
                    logits = outputs.logits
                    
                    # 获取预测的Token ID
                    predicted_token_ids = torch.argmax(logits, dim=-1)
                    
                    # 分析每个样本
                    batch_size = len(actual_input_ids)
                    
                    for i in range(min(batch_size, 1)):  # 只分析第一个样本
                        try:
                            # 获取样本数据
                            if isinstance(actual_input_ids, torch.Tensor):
                                sample_input = actual_input_ids[i].cpu().numpy()
                            else:
                                sample_input = actual_input_ids[i]
                            
                            if isinstance(labels, torch.Tensor):
                                sample_labels = labels[i].cpu().numpy()
                            else:
                                sample_labels = labels[i]
                            
                            if isinstance(predicted_token_ids, torch.Tensor):
                                sample_predictions = predicted_token_ids[i].cpu().numpy()
                            else:
                                sample_predictions = predicted_token_ids[i]
                            
                            # 找到要训练的部分
                            trainable_mask = sample_labels != -100
                            trainable_positions = np.where(trainable_mask)[0]
                            
                            if len(trainable_positions) > 0:
                                # 获取预测的Token（在可训练位置）
                                predicted_tokens = sample_predictions[trainable_mask]
                                target_tokens = sample_labels[trainable_mask]
                                
                                # 只在特定步骤记录详细信息
                                if self.step_count % self.log_interval == 0:
                                    self.loggers["labels"].info(f"\n🎯 样本 {i+1} 预测分析:")
                                    self.loggers["labels"].info(f"  📏 可训练位置数: {len(trainable_positions)}")
                                    
                                    # 显示预测的Token ID
                                    self.loggers["labels"].info(f"  🔮 模型预测Token ID: {predicted_tokens.tolist()}")
                                    self.loggers["labels"].info(f"  🎯 目标Token ID: {target_tokens.tolist()}")
                                    
                                    # 解码预测的Token
                                    predicted_text = self._decode_tokens(predicted_tokens.tolist())
                                    target_text = self._decode_tokens(target_tokens.tolist())
                                    
                                    self.loggers["labels"].info(f"  🔮 模型预测文本: {predicted_text}")
                                    self.loggers["labels"].info(f"  🎯 目标文本: {target_text}")
                                    
                                    # 分析预测文本的变化
                                    self._analyze_prediction_text_changes(predicted_text, i)
                                    
                                    # 计算预测准确率
                                    correct_predictions = np.sum(predicted_tokens == target_tokens)
                                    accuracy = correct_predictions / len(target_tokens) * 100
                                    self.loggers["labels"].info(f"  📊 预测准确率: {accuracy:.2f}% ({correct_predictions}/{len(target_tokens)})")
                                    
                                    # 调试信息：检查Token长度和类型
                                    self.loggers["labels"].info(f"  🔍 调试信息:")
                                    self.loggers["labels"].info(f"    预测Token长度: {len(predicted_tokens)}")
                                    self.loggers["labels"].info(f"    目标Token长度: {len(target_tokens)}")
                                    self.loggers["labels"].info(f"    预测Token类型: {type(predicted_tokens)}")
                                    self.loggers["labels"].info(f"    目标Token类型: {type(target_tokens)}")
                                    
                                    # 检查是否有任何匹配
                                    if correct_predictions == 0:
                                        self.loggers["labels"].info(f"  ⚠️ 警告: 预测准确率为0%，可能的原因:")
                                        self.loggers["labels"].info(f"    1. 模型在训练初期，还未学会正确预测")
                                        self.loggers["labels"].info(f"    2. 预测位置可能不正确")
                                        self.loggers["labels"].info(f"    3. Token对齐可能有问题")
                                        
                                        # 显示前5个Token的详细对比
                                        self.loggers["labels"].info(f"  🔍 前5个Token详细对比:")
                                        for j in range(min(5, len(predicted_tokens))):
                                            pred_token = predicted_tokens[j]
                                            target_token = target_tokens[j]
                                            pred_text = self._decode_tokens([pred_token])
                                            target_text_single = self._decode_tokens([target_token])
                                            self.loggers["labels"].info(f"    位置{j}: 预测:{pred_token}({pred_text}) vs 目标:{target_token}({target_text_single})")
                                    
                                    # 显示前10个位置的详细对比
                                    self.loggers["labels"].info(f"  🔍 前10个位置详细对比:")
                                    for j in range(min(10, len(predicted_tokens))):
                                        pred_token = predicted_tokens[j]
                                        target_token = target_tokens[j]
                                        pred_text = self._decode_tokens([pred_token])
                                        target_text_single = self._decode_tokens([target_token])
                                        match_symbol = "✅" if pred_token == target_token else "❌"
                                        self.loggers["labels"].info(f"    位置{j}: {match_symbol} 预测:{pred_token}({pred_text}) vs 目标:{target_token}({target_text_single})")
                                    
                                    # 分析-100部分（忽略的Token）
                                    ignore_mask = sample_labels == -100
                                    ignore_positions = np.where(ignore_mask)[0]
                                    ignore_tokens = sample_input[ignore_mask]
                                    
                                    if len(ignore_positions) > 0:
                                        self.loggers["labels"].info(f"\n🚫 忽略的Token分析 (-100部分):")
                                        self.loggers["labels"].info(f"  📏 忽略位置数: {len(ignore_positions)}")
                                        self.loggers["labels"].info(f"  📍 忽略位置: {ignore_positions.tolist()}")
                                        self.loggers["labels"].info(f"  🔤 忽略Token ID: {ignore_tokens.tolist()}")
                                        
                                        # 解码忽略的Token
                                        ignore_text = self._decode_tokens(ignore_tokens.tolist())
                                        self.loggers["labels"].info(f"  🔤 忽略Token文本: {ignore_text}")
                                        
                                        # 分析多轮对话结构
                                        self.loggers["labels"].info(f"\n💬 多轮对话结构分析:")
                                        self.loggers["labels"].info(f"  📊 总长度: {len(sample_input)}")
                                        self.loggers["labels"].info(f"  🎯 训练部分: {len(trainable_positions)} ({len(trainable_positions)/len(sample_input)*100:.1f}%)")
                                        self.loggers["labels"].info(f"  🚫 忽略部分: {len(ignore_positions)} ({len(ignore_positions)/len(sample_input)*100:.1f}%)")
                                        
                                        # 分析对话分段
                                        self._analyze_conversation_segments(sample_input, sample_labels, i)
                                
                                # 分析预测变化
                                if hasattr(self, 'previous_predictions') and i in self.previous_predictions:
                                    prev_predictions = self.previous_predictions[i]
                                    if len(prev_predictions) == len(predicted_tokens):
                                        changes = np.sum(prev_predictions != predicted_tokens)
                                        
                                        if changes > 0:
                                            change_positions = np.where(prev_predictions != predicted_tokens)[0]
                                            self.loggers["labels"].info(f"\n🔄 步骤 {self.step_count} 预测变化:")
                                            self.loggers["labels"].info(f"  📊 变化数量: {changes}/{len(predicted_tokens)}")
                                            self.loggers["labels"].info(f"  📍 变化位置: {change_positions.tolist()}")
                                            
                                            # 显示具体的变化
                                            for pos in change_positions:
                                                prev_token = prev_predictions[pos]
                                                curr_token = predicted_tokens[pos]
                                                target_token = target_tokens[pos]
                                                prev_text = self._decode_tokens([prev_token])
                                                curr_text = self._decode_tokens([curr_token])
                                                target_text = self._decode_tokens([target_token])
                                                self.loggers["labels"].info(f"    位置{pos}: {prev_token}({prev_text}) -> {curr_token}({curr_text}) [目标: {target_token}({target_text})]")
                                
                                # 保存当前预测用于下次比较
                                if not hasattr(self, 'previous_predictions'):
                                    self.previous_predictions = {}
                                self.previous_predictions[i] = predicted_tokens.copy()
                        
                        except Exception as e:
                            self.loggers["labels"].error(f"❌ 分析样本 {i} 预测失败: {e}")
                
                # 恢复训练模式
                model.train()
                
        except Exception as e:
            self.loggers["labels"].error(f"❌ 预测分析失败: {e}")
    
    def _analyze_conversation_segments(self, sample_input, sample_labels, sample_idx):
        """分析多轮对话的分段结构"""
        import numpy as np
        
        try:
            # 找到训练和忽略的分段
            trainable_mask = sample_labels != -100
            ignore_mask = sample_labels == -100
            
            # 找到分段的边界
            segments = []
            current_segment = []
            current_type = None
            
            for i, (is_trainable, token_id) in enumerate(zip(trainable_mask, sample_input)):
                segment_type = "trainable" if is_trainable else "ignore"
                
                if current_type != segment_type:
                    if current_segment:
                        segments.append({
                            'type': current_type,
                            'start': current_segment[0]['pos'],
                            'end': current_segment[-1]['pos'],
                            'length': len(current_segment),
                            'tokens': [item['token'] for item in current_segment]
                        })
                    current_segment = []
                    current_type = segment_type
                
                current_segment.append({
                    'pos': i,
                    'token': token_id
                })
            
            # 添加最后一个分段
            if current_segment:
                segments.append({
                    'type': current_type,
                    'start': current_segment[0]['pos'],
                    'end': current_segment[-1]['pos'],
                    'length': len(current_segment),
                    'tokens': [item['token'] for item in current_segment]
                })
            
            # 记录分段信息
            self.loggers["labels"].info(f"  📝 对话分段详情:")
            for seg_idx, segment in enumerate(segments):
                segment_text = self._decode_tokens(segment['tokens'])
                segment_type_emoji = "🎯" if segment['type'] == 'trainable' else "🚫"
                self.loggers["labels"].info(f"    分段{seg_idx+1}: {segment_type_emoji} {segment['type']} 位置{segment['start']}-{segment['end']} 长度{segment['length']}")
                self.loggers["labels"].info(f"      文本: {segment_text}")
            
            # 分析对话模式
            trainable_segments = [s for s in segments if s['type'] == 'trainable']
            ignore_segments = [s for s in segments if s['type'] == 'ignore']
            
            self.loggers["labels"].info(f"  📊 对话模式分析:")
            self.loggers["labels"].info(f"    训练分段数: {len(trainable_segments)}")
            self.loggers["labels"].info(f"    忽略分段数: {len(ignore_segments)}")
            
            if len(trainable_segments) > 1:
                self.loggers["labels"].info(f"    💬 多轮对话检测: 发现{len(trainable_segments)}个训练分段")
                for i, seg in enumerate(trainable_segments):
                    self.loggers["labels"].info(f"      轮次{i+1}: 位置{seg['start']}-{seg['end']} 长度{seg['length']}")
            
        except Exception as e:
            self.loggers["labels"].error(f"❌ 对话分段分析失败: {e}")
    
    def _analyze_prediction_text_changes(self, current_text, sample_idx):
        """分析预测文本的变化"""
        try:
            if hasattr(self, 'previous_prediction_texts') and sample_idx in self.previous_prediction_texts:
                previous_text = self.previous_prediction_texts[sample_idx]
                
                if previous_text != current_text:
                    self.loggers["labels"].info(f"\n📝 步骤 {self.step_count} 预测文本变化:")
                    self.loggers["labels"].info(f"  🔄 文本发生变化!")
                    
                    # 计算文本相似度
                    similarity = self._calculate_text_similarity(previous_text, current_text)
                    self.loggers["labels"].info(f"  📊 文本相似度: {similarity:.2f}%")
                    
                    # 显示变化的部分
                    self._show_text_differences(previous_text, current_text)
                else:
                    self.loggers["labels"].info(f"\n📝 步骤 {self.step_count} 预测文本变化:")
                    self.loggers["labels"].info(f"  ✅ 文本未发生变化")
            
            # 保存当前预测文本
            if not hasattr(self, 'previous_prediction_texts'):
                self.previous_prediction_texts = {}
            self.previous_prediction_texts[sample_idx] = current_text
            
        except Exception as e:
            self.loggers["labels"].error(f"❌ 预测文本变化分析失败: {e}")
    
    def _calculate_text_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        try:
            # 简单的字符级相似度计算
            if len(text1) == 0 and len(text2) == 0:
                return 100.0
            if len(text1) == 0 or len(text2) == 0:
                return 0.0
            
            # 使用编辑距离计算相似度
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, text1, text2).ratio()
            return similarity * 100
        except Exception:
            return 0.0
    
    def _show_text_differences(self, old_text, new_text):
        """显示文本差异"""
        try:
            from difflib import unified_diff
            
            self.loggers["labels"].info(f"  📋 文本变化详情:")
            self.loggers["labels"].info(f"    之前: {old_text}")
            self.loggers["labels"].info(f"    现在: {new_text}")
            
            # 使用unified_diff显示差异
            diff_lines = list(unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile='之前',
                tofile='现在',
                lineterm=''
            ))
            
            if diff_lines:
                self.loggers["labels"].info(f"  🔍 差异分析:")
                for line in diff_lines[:10]:  # 只显示前10行差异
                    if line.startswith('+'):
                        self.loggers["labels"].info(f"    ➕ 新增: {line[1:].strip()}")
                    elif line.startswith('-'):
                        self.loggers["labels"].info(f"    ➖ 删除: {line[1:].strip()}")
                    elif line.startswith('@@'):
                        self.loggers["labels"].info(f"    📍 {line.strip()}")
            
            # 分析变化类型
            self._analyze_change_types(old_text, new_text)
            
        except Exception as e:
            self.loggers["labels"].error(f"❌ 文本差异分析失败: {e}")
    
    def _analyze_change_types(self, old_text, new_text):
        """分析变化类型"""
        try:
            changes = {
                'added_chars': 0,
                'removed_chars': 0,
                'modified_chars': 0
            }
            
            # 简单的变化分析
            if len(new_text) > len(old_text):
                changes['added_chars'] = len(new_text) - len(old_text)
            elif len(new_text) < len(old_text):
                changes['removed_chars'] = len(old_text) - len(new_text)
            
            # 计算修改的字符数
            min_len = min(len(old_text), len(new_text))
            for i in range(min_len):
                if old_text[i] != new_text[i]:
                    changes['modified_chars'] += 1
            
            self.loggers["labels"].info(f"  📈 变化统计:")
            self.loggers["labels"].info(f"    新增字符: {changes['added_chars']}")
            self.loggers["labels"].info(f"    删除字符: {changes['removed_chars']}")
            self.loggers["labels"].info(f"    修改字符: {changes['modified_chars']}")
            
        except Exception as e:
            self.loggers["labels"].error(f"❌ 变化类型分析失败: {e}")
    
    @override
    def on_step_end(self, args, state, control, **kwargs):
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
            
            # 记录步骤信息
            from datetime import datetime
            step_info = {
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
                "loss": state.log_history[-1].get('loss') if state.log_history else None,
                "training_time": getattr(state, 'training_time', None)
            }
            
            self.training_history.append(step_info)
    
    @override
    def on_log(self, args, state, control, **kwargs):
        """在日志记录时调用，可以获取到训练数据"""
        if self.step_count % self.log_interval == 0:
            # 尝试获取当前批次的数据
            if hasattr(kwargs, 'logs') and kwargs['logs']:
                logs = kwargs['logs']
                self.loggers["labels"].info(f"📊 训练日志: {logs}")
    
    def analyze_training_tokens(self, model, inputs, labels):
        """分析训练过程中的token变化"""
        import numpy as np
        from datetime import datetime
        
        # 只在特定步骤记录详细信息
        if self.step_count % self.log_interval == 0:
            self.loggers["labels"].info(f"\n🔍 训练Token分析 - 步骤 {self.step_count}")
            self.loggers["labels"].info(f"{'='*80}")
        
        # 获取input_ids和labels
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
        else:
            input_ids = inputs
            attention_mask = None
        
        # 处理BatchEncoding对象
        if hasattr(input_ids, 'input_ids'):
            # 如果是BatchEncoding，获取实际的input_ids tensor
            actual_input_ids = input_ids.input_ids
        else:
            actual_input_ids = input_ids
        
        # 详细调试信息
        if self.step_count % self.log_interval == 0:
            self.loggers["labels"].info(f"🔍 详细调试信息:")
            self.loggers["labels"].info(f"  inputs类型: {type(inputs)}")
            if isinstance(inputs, dict):
                self.loggers["labels"].info(f"  inputs键: {list(inputs.keys())}")
                for key, value in inputs.items():
                    self.loggers["labels"].info(f"    {key}: 类型={type(value)}, 形状={getattr(value, 'shape', 'N/A')}")
            
            self.loggers["labels"].info(f"  input_ids类型: {type(input_ids)}")
            self.loggers["labels"].info(f"  actual_input_ids类型: {type(actual_input_ids)}")
            self.loggers["labels"].info(f"  labels类型: {type(labels)}")
            
            if actual_input_ids is not None:
                self.loggers["labels"].info(f"  actual_input_ids详细信息:")
                self.loggers["labels"].info(f"    类型: {type(actual_input_ids)}")
                self.loggers["labels"].info(f"    形状: {getattr(actual_input_ids, 'shape', 'N/A')}")
                self.loggers["labels"].info(f"    设备: {getattr(actual_input_ids, 'device', 'N/A')}")
                self.loggers["labels"].info(f"    数据类型: {getattr(actual_input_ids, 'dtype', 'N/A')}")
                
            if labels is not None:
                self.loggers["labels"].info(f"  labels详细信息:")
                self.loggers["labels"].info(f"    类型: {type(labels)}")
                self.loggers["labels"].info(f"    形状: {getattr(labels, 'shape', 'N/A')}")
                self.loggers["labels"].info(f"    设备: {getattr(labels, 'device', 'N/A')}")
                self.loggers["labels"].info(f"    数据类型: {getattr(labels, 'dtype', 'N/A')}")
                
            # 尝试不同的访问方式
            self.loggers["labels"].info(f"🔍 尝试访问方式:")
            try:
                if actual_input_ids is not None:
                    self.loggers["labels"].info(f"  actual_input_ids[0] 类型: {type(actual_input_ids[0])}")
                    if hasattr(actual_input_ids[0], 'shape'):
                        self.loggers["labels"].info(f"  actual_input_ids[0] 形状: {actual_input_ids[0].shape}")
            except Exception as e:
                self.loggers["labels"].error(f"  ❌ actual_input_ids[0] 访问失败: {e}")
                
            try:
                if labels is not None:
                    self.loggers["labels"].info(f"  labels[0] 类型: {type(labels[0])}")
                    if hasattr(labels[0], 'shape'):
                        self.loggers["labels"].info(f"  labels[0] 形状: {labels[0].shape}")
            except Exception as e:
                self.loggers["labels"].error(f"  ❌ labels[0] 访问失败: {e}")
        
        if actual_input_ids is not None and labels is not None:
            # 分析每个样本
            batch_size = len(actual_input_ids)
            
            for i in range(min(batch_size, 1)):  # 只分析第一个样本
                try:
                    # 使用正确的tensor索引方式
                    if isinstance(actual_input_ids, torch.Tensor):
                        sample_input = actual_input_ids[i].cpu().numpy()
                    else:
                        sample_input = actual_input_ids[i]
                    
                    if isinstance(labels, torch.Tensor):
                        sample_labels = labels[i].cpu().numpy()
                    else:
                        sample_labels = labels[i]
                except Exception as e:
                    self.loggers["labels"].error(f"❌ 访问样本 {i} 失败: {e}")
                    continue
                
                # 找到要训练的部分（labels != -100的部分）
                trainable_mask = sample_labels != -100
                trainable_positions = np.where(trainable_mask)[0]
                trainable_tokens = sample_labels[trainable_mask]
                
                # 只在特定步骤记录详细信息
                if self.step_count % self.log_interval == 0:
                    self.loggers["labels"].info(f"\n📝 样本 {i+1} 训练Token详情:")
                    self.loggers["labels"].info(f"  📏 总长度: {len(sample_input)}")
                    self.loggers["labels"].info(f"  🎯 可训练长度: {len(trainable_positions)}")
                    self.loggers["labels"].info(f"  📍 可训练位置: {trainable_positions.tolist()}")
                    self.loggers["labels"].info(f"  🔤 要训练的Token ID: {trainable_tokens.tolist()}")
                    
                    # 解码Token为中文
                    trainable_text = self._decode_tokens(trainable_tokens.tolist())
                    self.loggers["labels"].info(f"  🔤 要训练的Token文本: {trainable_text}")
                    
                    # 显示对应的input tokens
                    trainable_input_tokens = sample_input[trainable_mask]
                    self.loggers["labels"].info(f"  📥 对应的Input Token ID: {trainable_input_tokens.tolist()}")
                    
                    # 解码Input Token为中文
                    input_text = self._decode_tokens(trainable_input_tokens.tolist())
                    self.loggers["labels"].info(f"  📥 对应的Input Token文本: {input_text}")
                    
                    # 显示完整的input_ids和labels（只显示前100个和后100个，避免日志过长）
                    if len(sample_input) > 200:
                        input_preview = sample_input[:100].tolist() + ["..."] + sample_input[-100:].tolist()
                        labels_preview = sample_labels[:100].tolist() + ["..."] + sample_labels[-100:].tolist()
                    else:
                        input_preview = sample_input.tolist()
                        labels_preview = sample_labels.tolist()
                    
                    self.loggers["labels"].info(f"  📋 完整Input IDs (预览): {input_preview}")
                    self.loggers["labels"].info(f"  🏷️ 完整Labels (预览): {labels_preview}")
                
                # 分析token变化（每次都检查）
                if hasattr(self, 'previous_tokens') and i in self.previous_tokens:
                    prev_tokens = self.previous_tokens[i]
                    if len(prev_tokens) == len(trainable_tokens):
                        changes = np.sum(prev_tokens != trainable_tokens)
                        
                        if changes > 0:
                            change_positions = np.where(prev_tokens != trainable_tokens)[0]
                            self.loggers["labels"].info(f"\n🔄 步骤 {self.step_count} Token变化:")
                            self.loggers["labels"].info(f"  📊 变化数量: {changes}/{len(trainable_tokens)}")
                            self.loggers["labels"].info(f"  📍 变化位置: {change_positions.tolist()}")
                            
                            # 显示具体的变化（包含解码文本）
                            for pos in change_positions:
                                prev_token = prev_tokens[pos]
                                curr_token = trainable_tokens[pos]
                                prev_text = self._decode_tokens([prev_token])
                                curr_text = self._decode_tokens([curr_token])
                                self.loggers["labels"].info(f"    位置{pos}: {prev_token}({prev_text}) -> {curr_token}({curr_text})")
                
                # 保存当前tokens用于下次比较
                if not hasattr(self, 'previous_tokens'):
                    self.previous_tokens = {}
                self.previous_tokens[i] = trainable_tokens.copy()
    
    @override
    def on_evaluate(self, args, state, control, **kwargs):
        """在评估时调用"""
        self.loggers["predictions"].info(f"\n{'='*80}")
        self.loggers["predictions"].info(f"📊 评估阶段监控 - 步骤 {state.global_step}")
        self.loggers["predictions"].info(f"{'='*80}")
        
        # 如果有预测结果，进行分析
        if hasattr(kwargs, 'predict_results') and kwargs['predict_results'] is not None:
            self._analyze_predictions(kwargs['predict_results'], state.global_step)
    
    @override
    def on_predict(self, args, state, control, **kwargs):
        """在预测时调用"""
        self.loggers["predictions"].info(f"\n{'='*80}")
        self.loggers["predictions"].info(f"🔮 预测阶段监控 - 步骤 {state.global_step}")
        self.loggers["predictions"].info(f"{'='*80}")
        
        # 获取预测结果
        predict_results = kwargs.get('predict_results')
        if predict_results is not None:
            self._analyze_predictions(predict_results, state.global_step)
    
    def _analyze_predictions(self, predict_results, step: int):
        """分析预测结果"""
        import numpy as np
        from datetime import datetime
        
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
    
    def _remove_padding(self, tokens, pad_token_id: int = -100):
        """移除padding tokens"""
        import numpy as np
        # 找到非padding的位置
        non_pad_mask = tokens != pad_token_id
        if np.any(non_pad_mask):
            # 找到第一个和最后一个非padding位置
            first_non_pad = np.argmax(non_pad_mask)
            last_non_pad = len(tokens) - 1 - np.argmax(non_pad_mask[::-1])
            return tokens[first_non_pad:last_non_pad+1]
        else:
            return tokens
    
    def _analyze_alignment(self, predictions, labels):
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
    
    @override
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时调用"""
        from datetime import datetime
        
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