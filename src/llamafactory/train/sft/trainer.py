# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..token_loss_callback import TokenLossFinalizeCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ..token_loss_tracker import TokenLossTracker


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)
        # 添加标签预测监控回调
        from ..callbacks import LabelPredictionMonitorCallback
        monitor_callback = LabelPredictionMonitorCallback(
            output_dir=self.args.output_dir,
            log_interval=5,
            save_detailed_logs=True
        )
        self.add_callback(monitor_callback)
        
        # 设置tokenizer到监控回调
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            monitor_callback.set_tokenizer(self.tokenizer)
        elif hasattr(self, 'processing_class') and self.processing_class is not None:
            monitor_callback.set_tokenizer(self.processing_class)


        # 检查是否有use_dft_loss属性，如果没有则跳过
        if hasattr(finetuning_args, 'use_dft_loss') and finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func
        
        # 初始化Token-level Loss Tracker
        self.token_loss_tracker = TokenLossTracker(
            output_dir=self.args.output_dir,
            save_interval=100,  # Save every 100 records
            max_samples_per_step=1,  # Track 1 sample per step to save memory
            context_window_size=10,  # Context window size (left and right)
        )
        
        # Set tokenizer for tracker
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.token_loss_tracker.set_tokenizer(self.tokenizer)
        elif hasattr(self, 'processing_class') and self.processing_class is not None:
            self.token_loss_tracker.set_tokenizer(self.processing_class)
        
        # Add callback to finalize token loss tracking at the end of training
        self.add_callback(TokenLossFinalizeCallback())

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        # 检查是否在训练模式（评估时不应记录token-level loss）
        is_training = model.training
        
        # 调用监控回调分析token（仅在训练时调用）
        if is_training:
            for callback in self.callback_handler.callbacks:
                if hasattr(callback, 'analyze_training_tokens'):
                    try:
                        labels = inputs.get('labels')
                        if labels is not None:
                            callback.analyze_training_tokens(model, inputs, labels)
                    except Exception as e:
                        logger.warning(f"⚠️ Token分析失败: {e}")
                
                # 调用预测分析（仅在训练时调用）
                if hasattr(callback, 'analyze_model_predictions'):
                    try:
                        labels = inputs.get('labels')
                        if labels is not None:
                            callback.analyze_model_predictions(model, inputs, labels)
                    except Exception as e:
                        logger.warning(f"⚠️ 预测分析失败: {e}")
        
        # 获取模型输出（用于计算loss和记录token-level loss）
        labels = inputs.get('labels')
        outputs = None  # Initialize outputs to None
        
        if labels is not None:
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 记录token-level loss（仅在训练时记录，使用detach避免影响梯度）
            if is_training:
                try:
                    self.token_loss_tracker.record_token_losses(
                        logits=logits.detach(),  # Detach to avoid affecting gradients
                        labels=labels,
                        input_ids=inputs.get('input_ids'),
                        step=self.state.global_step if hasattr(self, 'state') else None,
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Token-level loss记录失败: {e}")
            
            # 计算标准loss（用于训练和评估）
            if hasattr(self, 'compute_loss_func') and self.compute_loss_func is not None:
                loss = self.compute_loss_func(model, inputs, outputs, return_outputs=False)
            else:
                # Use default loss computation
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # Fallback to default behavior
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, *args, **kwargs)
            if return_outputs:
                loss, outputs = result
            else:
                loss = result
        
        # 根据return_outputs参数决定返回值
        if return_outputs:
            return loss, outputs
        else:
            return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
    
    def _finalize_token_loss_tracking(self):
        """Finalize token loss tracking and save remaining data."""
        if hasattr(self, 'token_loss_tracker'):
            try:
                self.token_loss_tracker.finalize()
                logger.info_rank0("Token loss tracking finalized.")
            except Exception as e:
                logger.warning(f"Failed to finalize token loss tracking: {e}")
