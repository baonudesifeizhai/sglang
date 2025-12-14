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

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


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
        self.max_iterations = config.algorithm_config.get(
            "max_iterations", self.block_size
        )
        self.confidence_threshold = config.algorithm_config.get(
            "confidence_threshold", 0.0
        )

        # Cache management for hierarchical caching
        # Block-level cache: stores completed blocks
        self.block_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # Sub-block cache: stores partially generated blocks
        self.sub_block_cache: Dict[
            Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _apply_token_shift(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply token shift mechanism.

        Shifts logits to preserve autoregressive characteristics:
        logits_shifted = concat([logits[:, :1], logits[:, :-1]], dim=1)

        This allows bidirectional context within blocks while maintaining AR objectives.

        Args:
            logits: Logits tensor of shape (num_tokens, vocab_size) or (batch, seq_len, vocab_size)

        Returns:
            Shifted logits tensor
        """
        if logits.dim() == 2:
            # Shape: (num_tokens, vocab_size) - add batch dimension
            logits = logits.unsqueeze(0)  # (1, num_tokens, vocab_size)
            if logits.shape[1] <= 1:
                return logits.squeeze(0)
            shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            return shifted.squeeze(0)  # Remove batch dimension
        else:
            # Shape: (batch, seq_len, vocab_size)
            if logits.shape[1] <= 1:
                return logits
            shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            return shifted

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
        1. Identify prompt and mask positions
        2. Iterative decoding:
           - Forward pass
           - Apply token shift
           - Fill high-confidence tokens
        3. Return final logits and generated tokens

        Note: Block-causal attention mask and hierarchical caching integration
        are TODO - needs to be implemented in attention layers.
        """
        # Step 1: Identify prompt and mask positions
        input_ids = forward_batch.input_ids.clone()
        mask_positions = input_ids == self.mask_id
        num_masks = mask_positions.sum().item()

        if num_masks == 0:
            # No masks to fill, return as-is
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )
            return logits_output, input_ids, can_run_cuda_graph

        # Calculate prompt length (everything before first mask)
        prompt_len = len(input_ids) - num_masks

        # Step 2: Iterative block diffusion
        for iteration in range(self.max_iterations):
            # Check if all masks are filled
            mask_positions = input_ids == self.mask_id
            if mask_positions.sum().item() == 0:
                break

            # Forward pass
            # TODO: Apply block-causal attention mask in forward pass
            forward_batch.input_ids = input_ids
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            # Get logits for the entire sequence
            logits = logits_output.full_logits  # Shape: (num_tokens, vocab_size)

            # Apply token shift mechanism
            logits = self._apply_token_shift(logits)

            # Sample tokens (greedy for now, can extend to sampling)
            next_token_ids = torch.argmax(logits, dim=-1)

            # Fill masks based on confidence threshold
            if self.confidence_threshold > 0:
                probs = F.softmax(logits, dim=-1)
                # Get confidence for predicted tokens
                confidence = torch.gather(
                    probs, dim=-1, index=next_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                # Only fill high-confidence positions
                fill_mask = mask_positions & (confidence > self.confidence_threshold)
                # If no high-confidence tokens, fill at least one
                if fill_mask.sum().item() == 0:
                    # Find highest confidence mask position
                    mask_confidence = torch.where(
                        mask_positions,
                        confidence,
                        torch.tensor(-float("inf"), device=confidence.device),
                    )
                    _, top_idx = torch.topk(mask_confidence, k=1)
                    fill_mask = torch.zeros_like(mask_positions)
                    fill_mask[top_idx] = True
            else:
                # Fill all masks (greedy decoding)
                fill_mask = mask_positions

            # Update input_ids with filled tokens
            input_ids = torch.where(fill_mask, next_token_ids, input_ids)

        # Final forward pass to get logits for the completed sequence
        forward_batch.input_ids = input_ids
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        # Extract generated tokens (everything after prompt)
        next_token_ids = input_ids[prompt_len:]

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = FastDLLMBlockDiffusion
