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

"""
Token-level Loss Tracker for SFT Training Analysis

This module provides functionality to track and analyze per-token losses during training,
enabling detailed analysis of which tokens are problematic.
"""

import json
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

from ..extras import logging
from ..extras.constants import IGNORE_INDEX

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = logging.get_logger(__name__)


class TokenLossTracker:
    """Track and save token-level losses during training."""
    
    def __init__(
        self,
        output_dir: str,
        tokenizer: Optional["PreTrainedTokenizer"] = None,
        save_interval: int = 100,
        max_samples_per_step: int = 1,
        context_window_size: int = 10,
    ):
        """
        Initialize the token loss tracker.
        
        Args:
            output_dir: Directory to save token loss data
            tokenizer: Tokenizer for decoding tokens
            save_interval: Save data every N steps
            max_samples_per_step: Maximum number of samples to track per step
            context_window_size: Number of tokens to include on each side (left and right) for context
        """
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.save_interval = save_interval
        self.max_samples_per_step = max_samples_per_step
        self.context_window_size = context_window_size
        
        # Create output directory
        self.token_loss_dir = os.path.join(output_dir, "token_loss_data")
        os.makedirs(self.token_loss_dir, exist_ok=True)
        
        # Storage for token loss data
        self.token_loss_records = []
        self.step_count = 0
        
        # Statistics
        self.token_loss_stats = defaultdict(list)  # token_id -> [losses]
        self.position_loss_stats = defaultdict(list)  # position -> [losses]
        
        logger.info_rank0(f"TokenLossTracker initialized. Output dir: {self.token_loss_dir}")
    
    def set_tokenizer(self, tokenizer: "PreTrainedTokenizer"):
        """Set the tokenizer for decoding tokens."""
        self.tokenizer = tokenizer
        logger.info_rank0(f"Tokenizer set: {type(tokenizer).__name__}")
    
    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        if self.tokenizer is None:
            return f"<token_{token_id}>"
        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            return f"<decode_error_{token_id}>"
    
    def _extract_context_window(
        self,
        pos: int,
        sample_input: Optional[np.ndarray],
        sample_labels: np.ndarray,
        sample_losses: np.ndarray,
        window_size: int,
    ) -> Dict[str, Any]:
        """
        Extract context window around a token position.
        
        Args:
            pos: Position in shift_labels (0-indexed, after shift)
            sample_input: Original input_ids [seq_len] (if available)
            sample_labels: Shifted labels [seq_len-1]
            sample_losses: Token losses [seq_len-1]
            window_size: Number of tokens on each side
        
        Returns:
            Dictionary with context window information
        """
        context_info = {
            "window_size": window_size,
            "left_context_tokens": [],
            "right_context_tokens": [],
            "left_context_losses": [],
            "right_context_losses": [],
        }
        
        # Calculate context positions
        # pos is in shift_labels space (0 to seq_len-2)
        # For context, we need to map back to input_ids space
        # shift_labels[pos] corresponds to input_ids[pos+1]
        # So context center in input_ids is pos+1
        
        if sample_input is not None:
            # Use input_ids for context tokens
            # pos is in shift_labels space: shift_labels[pos] corresponds to input_ids[pos+1]
            # So the token being predicted is at input_ids[pos+1]
            input_len = len(sample_input)
            center_pos_in_input = pos + 1  # Map shift_labels position to input_ids position
            
            # Left context: tokens before the current token
            # [center_pos_in_input - window_size, center_pos_in_input)
            left_start = max(0, center_pos_in_input - window_size)
            left_end = center_pos_in_input
            left_context_ids = sample_input[left_start:left_end].tolist()
            left_context_tokens = [self._decode_token(tid) for tid in left_context_ids]
            
            # Right context: tokens after the current token
            # (center_pos_in_input, center_pos_in_input + window_size]
            right_start = center_pos_in_input + 1
            right_end = min(input_len, center_pos_in_input + 1 + window_size)
            right_context_ids = sample_input[right_start:right_end].tolist()
            right_context_tokens = [self._decode_token(tid) for tid in right_context_ids]
            
            context_info["left_context_tokens"] = left_context_tokens
            context_info["right_context_tokens"] = right_context_tokens
        else:
            # Fallback: use labels for context (less accurate but better than nothing)
            seq_len = len(sample_labels)
            
            # Left context: [pos - window_size, pos)
            left_start = max(0, pos - window_size)
            left_end = pos
            left_context_ids = sample_labels[left_start:left_end].tolist()
            left_context_tokens = [
                self._decode_token(tid) if tid != IGNORE_INDEX else "<IGNORE>"
                for tid in left_context_ids
            ]
            
            # Right context: (pos, pos + window_size]
            right_start = pos + 1
            right_end = min(seq_len, pos + 1 + window_size)
            right_context_ids = sample_labels[right_start:right_end].tolist()
            right_context_tokens = [
                self._decode_token(tid) if tid != IGNORE_INDEX else "<IGNORE>"
                for tid in right_context_ids
            ]
            
            context_info["left_context_tokens"] = left_context_tokens
            context_info["right_context_tokens"] = right_context_tokens
        
        # Extract losses for context tokens
        # Left context losses: positions [pos - window_size, pos) in shift_labels
        left_loss_start = max(0, pos - window_size)
        left_loss_end = pos
        if left_loss_start < left_loss_end:
            left_context_losses = sample_losses[left_loss_start:left_loss_end].tolist()
        else:
            left_context_losses = []
        
        # Right context losses: positions (pos, pos + window_size] in shift_labels
        right_loss_start = pos + 1
        right_loss_end = min(len(sample_losses), pos + 1 + window_size)
        if right_loss_start < right_loss_end:
            right_context_losses = sample_losses[right_loss_start:right_loss_end].tolist()
        else:
            right_context_losses = []
        
        context_info["left_context_losses"] = left_context_losses
        context_info["right_context_losses"] = right_context_losses
        
        return context_info
    
    def _classify_token_type(self, token_str: str, token_id: int) -> str:
        """
        Classify token into categories for analysis.
        
        Returns:
            Token type: 'structural', 'keyword', 'numeric', 'path', 'natural_language', 'unknown'
        """
        # Structural tokens
        structural_chars = {'{', '}', '[', ']', '<', '>', '/', ',', ':', ';', '(', ')', '"', "'"}
        if any(char in token_str for char in structural_chars) or token_str.strip() in structural_chars:
            return "structural"
        
        # XML/HTML tags
        if token_str.startswith('</') or token_str.startswith('<') and token_str.endswith('>'):
            return "structural"
        
        # Keywords (common programming/format keywords)
        keywords = {'insert_after', 'anchor', 'function', 'method', 'class', 'def', 'return', 'if', 'else', 'for', 'while'}
        if token_str.lower() in keywords:
            return "keyword"
        
        # Numeric patterns
        if token_str.replace('.', '').replace('-', '').isdigit() or token_str.startswith('0x'):
            return "numeric"
        
        # Path patterns
        if '/' in token_str or '\\' in token_str or token_str.endswith('.xml') or token_str.endswith('.json'):
            return "path"
        
        # Natural language (default for most tokens)
        return "natural_language"
    
    def record_token_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ):
        """
        Record token-level losses for a batch.
        
        Args:
            logits: Model logits, shape [batch_size, seq_len, vocab_size]
            labels: Ground truth labels, shape [batch_size, seq_len]
            input_ids: Input token IDs, shape [batch_size, seq_len]
            step: Current training step
        """
        if step is not None:
            self.step_count = step
        
        # Compute per-token loss using reduction="none"
        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            reduction="none"
        )
        
        # Reshape for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Compute per-token losses
        token_losses = loss_fct(flat_logits, flat_labels)
        
        # Reshape back to [batch_size, seq_len-1]
        batch_size, seq_len = shift_labels.shape
        token_losses = token_losses.view(batch_size, seq_len)
        
        # Get probabilities and top-k predictions
        probs = torch.softmax(shift_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)
        
        # Get input_ids for context window (if provided)
        # Note: input_ids should be the original input, not shifted
        # We need to align it with shift_labels (which is labels shifted by 1)
        if input_ids is not None:
            # input_ids is [batch, seq_len], shift_labels is [batch, seq_len-1]
            # For position pos in shift_labels, the corresponding input_ids position is pos+1
            # But we want the context around the token being predicted, so we use pos+1 as center
            sample_input_ids = input_ids.detach().cpu().numpy()
        else:
            # If input_ids not provided, we can't get context, but still record basic info
            sample_input_ids = None
        
        # Process each sample in the batch
        num_samples = min(batch_size, self.max_samples_per_step)
        
        for b in range(num_samples):
            sample_losses = token_losses[b].detach().cpu().numpy()
            sample_labels = shift_labels[b].detach().cpu().numpy()
            sample_topk_probs = topk_probs[b].detach().cpu().numpy()
            sample_topk_indices = topk_indices[b].detach().cpu().numpy()
            
            # Get input_ids for this sample if available
            sample_input = sample_input_ids[b] if sample_input_ids is not None else None
            
            # Only record non-ignored tokens
            valid_mask = sample_labels != IGNORE_INDEX
            valid_positions = np.where(valid_mask)[0]
            
            if len(valid_positions) == 0:
                continue
            
            # Record each valid token
            for pos_idx, pos in enumerate(valid_positions):
                token_id = int(sample_labels[pos])
                token_loss = float(sample_losses[pos])
                token_str = self._decode_token(token_id)
                token_type = self._classify_token_type(token_str, token_id)
                
                # Get top-k predictions
                topk_tokens = [
                    {
                        "token_id": int(sample_topk_indices[pos, k]),
                        "token": self._decode_token(int(sample_topk_indices[pos, k])),
                        "prob": float(sample_topk_probs[pos, k])
                    }
                    for k in range(5)
                ]
                
                # Check if prediction is correct
                predicted_token_id = int(sample_topk_indices[pos, 0])
                is_correct = predicted_token_id == token_id
                
                # Extract context window
                context_info = self._extract_context_window(
                    pos=pos,
                    sample_input=sample_input,
                    sample_labels=sample_labels,
                    sample_losses=sample_losses,
                    window_size=self.context_window_size,
                )
                
                record = {
                    "step": self.step_count,
                    "sample_id": b,
                    "position": int(pos),
                    "gt_token_id": token_id,  # Renamed for clarity
                    "gt_token": token_str,     # Renamed for clarity
                    "gt_token_loss": token_loss,  # Renamed for clarity
                    "token_type": token_type,
                    "is_correct": is_correct,
                    "top1_pred_token": topk_tokens[0]["token"] if len(topk_tokens) > 0 else "",
                    "top1_pred_prob": topk_tokens[0]["prob"] if len(topk_tokens) > 0 else 0.0,
                    "topk_predictions": topk_tokens,
                    **context_info,  # Add context window information
                }
                
                self.token_loss_records.append(record)
                
                # Update statistics
                self.token_loss_stats[token_id].append(token_loss)
                self.position_loss_stats[pos].append(token_loss)
        
        # Save periodically
        if len(self.token_loss_records) >= self.save_interval:
            self._save_data()
    
    def _save_data(self):
        """Save accumulated token loss data to disk."""
        if not self.token_loss_records:
            return
        
        # Save detailed records
        filename = os.path.join(
            self.token_loss_dir,
            f"token_losses_step_{self.step_count:06d}.jsonl"
        )
        
        with open(filename, "w", encoding="utf-8") as f:
            for record in self.token_loss_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info_rank0(
            f"Saved {len(self.token_loss_records)} token loss records to {filename}"
        )
        
        # Clear records
        self.token_loss_records = []
    
    def finalize(self):
        """Save any remaining data and generate summary statistics."""
        if self.token_loss_records:
            self._save_data()
        
        # Save summary statistics
        summary_file = os.path.join(self.token_loss_dir, "token_loss_summary.json")
        
        # Compute token-level statistics
        token_stats = {}
        for token_id, losses in self.token_loss_stats.items():
            token_str = self._decode_token(token_id)
            token_type = self._classify_token_type(token_str, token_id)
            
            token_stats[token_id] = {
                "token": token_str,
                "token_type": token_type,
                "count": len(losses),
                "avg_loss": float(np.mean(losses)),
                "std_loss": float(np.std(losses)),
                "min_loss": float(np.min(losses)),
                "max_loss": float(np.max(losses)),
            }
        
        # Compute position-level statistics
        position_stats = {}
        for position, losses in self.position_loss_stats.items():
            position_stats[position] = {
                "count": len(losses),
                "avg_loss": float(np.mean(losses)),
                "std_loss": float(np.std(losses)),
                "min_loss": float(np.min(losses)),
                "max_loss": float(np.max(losses)),
            }
        
        summary = {
            "total_steps": self.step_count,
            "token_statistics": token_stats,
            "position_statistics": position_stats,
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info_rank0(f"Saved summary statistics to {summary_file}")
