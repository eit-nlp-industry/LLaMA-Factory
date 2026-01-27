# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

"""Callback to finalize token loss tracking at the end of training."""

from typing import TYPE_CHECKING

from transformers import TrainerCallback
from typing_extensions import override

from ..extras import logging

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = logging.get_logger(__name__)


class TokenLossFinalizeCallback(TrainerCallback):
    """Callback to finalize token loss tracking at the end of training."""
    
    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Finalize token loss tracking when training ends."""
        trainer = kwargs.get("trainer")
        if trainer is not None and hasattr(trainer, "token_loss_tracker"):
            try:
                trainer.token_loss_tracker.finalize()
                logger.info_rank0("✅ Token loss tracking finalized successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to finalize token loss tracking: {e}")
