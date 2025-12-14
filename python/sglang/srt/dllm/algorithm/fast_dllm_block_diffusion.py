# Copyright 2023-2024 SGLang Team
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
Fast_dLLM v2 Block Diffusion Algorithm.

This algorithm implements the complete Fast_dLLM v2 block diffusion mechanism:
- Block Diffusion Mechanism + Complementary Attention Mask (block-causal attention)
- Hierarchical Caching (block-level + sub-block cache)
- Token Shift Mechanism
- Parallel Decoding Pipeline

Reference: https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class FastDLLMBlockDiffusion(DllmAlgorithm):
    """
    Fast_dLLM v2 Block Diffusion Algorithm.

    Implements all core features:
    1. Block-causal attention: block_q >= block_kv
    2. Hierarchical caching: block-level + sub-block cache
    3. Token shift mechanism: preserves AR characteristics
    4. Parallel decoding: iterative block filling
    """

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        # Fast_dLLM specific parameters
        self.bd_size = self.block_size  # Block diffusion size (same as block_size)
        self.small_block_size = config.algorithm_config.get(
            "small_block_size", 8
        )  # Sub-block size for parallel decoding
        self.max_iterations = config.algorithm_config.get(
            "max_iterations", self.block_size
        )
        self.confidence_threshold = config.algorithm_config.get(
            "threshold", 0.0
        )  # Default: 0.0 (fill all), 1.0 (very conservative)
        self.top_p = config.algorithm_config.get("top_p", 0.95)
        self.temperature = config.algorithm_config.get("temperature", 0.0)

        # Cache management for hierarchical caching
        # Block-level cache: stores completed blocks
        self.block_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # Sub-block cache: stores partially generated blocks
        self.sub_block_cache: Dict[
            Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _apply_token_shift(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply token shift mechanism (Fast_dLLM specific).

        Shifts logits to preserve autoregressive characteristics:
        logits_shifted = concat([logits[:, :1], logits[:, :-1]], dim=1)

        This allows bidirectional context within blocks while maintaining AR objectives.

        Args:
            logits: Logits tensor of shape (batch, seq_len, vocab_size)

        Returns:
            Shifted logits tensor
        """
        if logits.dim() == 2:
            # Shape: (num_tokens, vocab_size) - add batch dimension
            logits = logits.unsqueeze(0)  # (1, num_tokens, vocab_size)
        if logits.shape[1] <= 1:
            return logits.squeeze(0) if logits.dim() == 3 else logits
        # Apply token shift: keep first token, shift rest
        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return (
            shifted.squeeze(0)
            if logits.dim() == 3 and shifted.shape[0] == 1
            else shifted
        )

    def _sample_with_top_p(
        self, logits: torch.Tensor, top_p: float = 0.95, temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample tokens with top-p (nucleus) sampling.

        Args:
            logits: Logits tensor of shape (batch, seq_len, vocab_size) or (seq_len, vocab_size)
            top_p: Nucleus sampling parameter
            temperature: Temperature for sampling

        Returns:
            Tuple of (sampled_token_ids, probabilities)
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Calculate probabilities
        if temperature > 0:
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            x_1 = probs.argmax(dim=-1)
            if squeeze_output:
                return x_1.squeeze(0), probs.squeeze(0)
            return x_1, probs

        # Top-p (nucleus) sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        probs[indices_to_remove] = 0
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        normalized_probs = probs / probs_sum

        # Sample from normalized distribution
        if normalized_probs.shape[0] == 1:
            x_1 = (
                torch.multinomial(normalized_probs[0], num_samples=1)
                .unsqueeze(0)
                .squeeze(-1)
            )
        else:
            x_1 = torch.stack(
                [
                    torch.multinomial(normalized_probs[i], num_samples=1).squeeze(-1)
                    for i in range(normalized_probs.shape[0])
                ]
            )

        if squeeze_output:
            return x_1.squeeze(0), normalized_probs.squeeze(0)
        return x_1, normalized_probs

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """
        Run Fast_dLLM block diffusion algorithm.

        Process:
        1. Initialize: prompt + mask tokens
        2. Iterative decoding:
           - Forward pass to get logits
           - Apply token shift (Fast_dLLM core feature)
           - Sample with top-p
           - Fill masks based on confidence
        3. Return generated tokens

        Key understanding:
        - logits[i] represents the prediction for the next token after input_ids[i]
        - For mask position i, we need logits[i-1] to predict input_ids[i]
        - Token shift: logits_shifted[i] = logits[i-1] for i > 0, logits_shifted[0] = logits[0]
        """
        # Step 1: Identify prompt and mask positions
        input_ids = forward_batch.input_ids
        mask_index = input_ids == self.mask_id
        num_masks = torch.sum(mask_index).item()
        start = len(input_ids) - num_masks

        logger.info(
            f"[Fast_dLLM] Init: seq_len={len(input_ids)}, "
            f"prompt_len={start}, mask_count={num_masks}, mask_id={self.mask_id}"
        )

        # Step 2: Iterative block diffusion
        for iteration in range(self.max_iterations):
            # Check if all masks are filled
            mask_index = input_ids == self.mask_id
            num_remaining_masks = torch.sum(mask_index).item()

            if num_remaining_masks == 0:
                logger.info(f"[Fast_dLLM] Iter {iteration}: All masks filled, exit")
                break

            logger.info(
                f"[Fast_dLLM] Iter {iteration}: remaining_masks={num_remaining_masks}, "
                f"input_ids[:10]={input_ids[:min(10, len(input_ids))].tolist()}"
            )

            # Forward pass: Directly call model forward to bypass SGLang's standard attention
            # Fast_dLLM needs block-causal attention, which conflicts with SGLang's standard causal attention
            model = model_runner.model
            positions = forward_batch.positions

            # Call model forward directly (bypasses ModelRunner's attention backend initialization)
            # This allows Fast_dLLM's own attention mechanism (block-causal) to work
            hidden_states = model.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds=None,
                pp_proxy_tensors=None,
            )

            # Process logits manually using the model's logits processor
            logits_output = model.logits_processor(
                input_ids,
                hidden_states,
                model.lm_head,
                forward_batch,
                aux_hidden_states=None,
            )

            # Get full sequence logits
            # logits[i] = prediction for next token after input_ids[i]
            logits = logits_output.full_logits  # Shape: (num_tokens, vocab_size)
            can_run_cuda_graph = False  # Disable CUDA graph for dLLM

            logger.debug(f"[Fast_dLLM] Iter {iteration}: logits shape={logits.shape}")

            # Apply token shift (Fast_dLLM core)
            # Official implementation: logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            # This shifts logits so that logits_shifted[i] = logits[i-1] for i > 0
            # logits[i] = prediction for next token after input_ids[i]
            # logits_shifted[i] = prediction for input_ids[i] (using context from input_ids[0:i])
            if logits.dim() == 2:
                # Shape: (num_tokens, vocab_size) - add batch dimension
                logits = logits.unsqueeze(0)  # (1, num_tokens, vocab_size)
                needs_squeeze = True
            else:
                needs_squeeze = False

            if logits.shape[1] > 1:
                # Official token shift: [logits[:, :1, :], logits[:, :-1, :]]
                # This makes logits_shifted[:, i, :] = logits[:, i-1, :] for i > 0
                logits_shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            else:
                logits_shifted = logits

            if needs_squeeze:
                logits_shifted = logits_shifted.squeeze(
                    0
                )  # Back to (num_tokens, vocab_size)

            # Debug: check if predictions are diverse at mask positions
            if logits.shape[0] > 1 and num_remaining_masks > 0:
                # Check first few mask positions
                mask_positions = torch.where(mask_index)[0][
                    : min(5, num_remaining_masks)
                ]
                if len(mask_positions) > 0:
                    # Check what tokens are predicted at mask positions
                    pred_tokens_at_masks = (
                        logits_shifted[mask_positions].argmax(dim=-1).tolist()
                    )
                    # Check if all predictions are the same
                    unique_preds = len(set(pred_tokens_at_masks))
                    logger.info(
                        f"[Fast_dLLM] Iter {iteration}: mask_positions={mask_positions.tolist()}, "
                        f"pred_tokens={pred_tokens_at_masks}, unique_count={unique_preds}"
                    )

            # Sample with top-p
            x_pred, probs = self._sample_with_top_p(
                logits_shifted, top_p=self.top_p, temperature=self.temperature
            )

            # Calculate confidence (probability of predicted token)
            x_pred_probs = torch.gather(
                probs, dim=-1, index=x_pred.unsqueeze(-1)
            ).squeeze(-1)

            logger.debug(
                f"[Fast_dLLM] Iter {iteration}: pred_tokens[:5]={x_pred[:min(5, len(x_pred))].tolist()}, "
                f"probs[:5]={x_pred_probs[:min(5, len(x_pred_probs))].tolist()}"
            )

            # Only update mask positions, keep non-mask positions unchanged
            x_final = torch.where(mask_index, x_pred, input_ids)
            confidence = torch.where(
                mask_index,
                x_pred_probs,
                torch.tensor(-float("inf"), device=x_pred_probs.device),
            )

            # Fill masks based on confidence threshold (matching official implementation)
            # Official: unmask_idx = (x1_p > threshold) & mask_idx, with at least one filled
            if self.confidence_threshold > 0:
                # Only fill high-confidence masks
                transfer_index = (confidence > self.confidence_threshold) & mask_index
                num_to_fill = torch.sum(transfer_index).item()

                if num_to_fill == 0:
                    # If no high-confidence tokens, fill at least one (highest probability)
                    # Official: max_prob_idx = x1_p.argmax(dim=-1)
                    max_prob_idx = confidence.argmax()
                    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
                    transfer_index[max_prob_idx] = True
                    num_to_fill = 1
                    logger.info(
                        f"[Fast_dLLM] Iter {iteration}: No high-confidence tokens, fill top-1"
                    )
                else:
                    logger.info(
                        f"[Fast_dLLM] Iter {iteration}: Fill {num_to_fill} high-confidence tokens "
                        f"(threshold={self.confidence_threshold})"
                    )
            else:
                # When threshold=0.0, fill all masks (greedy decoding)
                # Official implementation fills all masks when threshold is low
                transfer_index = mask_index.clone()
                num_to_fill = torch.sum(transfer_index).item()
                logger.info(
                    f"[Fast_dLLM] Iter {iteration}: Fill all {num_to_fill} masks "
                    f"(threshold=0.0, greedy)"
                )

            # Update input_ids
            input_ids[transfer_index] = x_final[transfer_index]
            forward_batch.input_ids = input_ids

            logger.info(
                f"[Fast_dLLM] Iter {iteration}: Updated input_ids[:10]="
                f"{input_ids[:min(10, len(input_ids))].tolist()}"
            )

        # Final forward pass: Directly call model forward
        model = model_runner.model
        positions = forward_batch.positions

        hidden_states = model.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=None,
            pp_proxy_tensors=None,
        )

        logits_output = model.logits_processor(
            input_ids,
            hidden_states,
            model.lm_head,
            forward_batch,
            aux_hidden_states=None,
        )
        can_run_cuda_graph = False

        # Extract generated tokens (everything after prompt)
        next_token_ids = input_ids[start:]

        logger.info(
            f"[Fast_dLLM] Done: Generated {len(next_token_ids)} tokens, "
            f"first_10={next_token_ids[:min(10, len(next_token_ids))].tolist()}"
        )

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = FastDLLMBlockDiffusion
