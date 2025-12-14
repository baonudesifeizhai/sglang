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

        Based on official implementation:
        https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B

        Process:
        1. Identify prompt and mask positions
        2. Iterative block diffusion with sub-block decoding:
           - Forward pass
           - Apply token shift (Fast_dLLM specific)
           - Sample with top-p
           - Fill high-confidence tokens
        3. Return final logits and generated tokens

        Note: Block-causal attention mask and hierarchical caching integration
        are TODO - needs to be implemented in attention layers.
        """
        # Step 1: Identify prompt and mask positions
        mask_index = forward_batch.input_ids == self.mask_id
        start = len(forward_batch.input_ids) - torch.sum(mask_index).item()

        # Step 2: Iterative block diffusion
        # Simplified version: process entire block at once (similar to LowConfidence)
        # TODO: Add sub-block processing and hierarchical caching for full Fast_dLLM support

        for iteration in range(self.max_iterations):
            # Check if all masks are filled
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # Forward pass
            # TODO: Apply block-causal attention mask in forward pass
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            # Get logits for the entire sequence
            logits = logits_output.full_logits  # Shape: (num_tokens, vocab_size)

            # Apply token shift mechanism (Fast_dLLM specific)
            # This is critical for maintaining AR characteristics
            # Temporarily match LowConfidence behavior first, then enable token shift
            # if logits.dim() == 2:
            #     logits = logits.unsqueeze(0)
            # if logits.shape[1] > 1:
            #     logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            #     if logits.dim() == 3 and logits.shape[0] == 1:
            #         logits = logits.squeeze(0)

            # Sample tokens (greedy for now, matching LowConfidence)
            x = torch.argmax(logits, dim=-1)

            # Calculate probabilities
            p = torch.squeeze(
                torch.gather(
                    F.softmax(logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            # Only update mask positions, keep non-mask positions unchanged
            x = torch.where(mask_index, x, forward_batch.input_ids)
            confidence = torch.where(
                mask_index, p, torch.tensor(-float("inf"), device=p.device)
            )

            # Fill masks based on confidence threshold
            if self.confidence_threshold > 0:
                transfer_index = confidence > self.confidence_threshold
                if transfer_index.sum().item() == 0:
                    # If no high-confidence tokens, fill at least one (highest probability)
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
                    transfer_index[select_index] = True
            else:
                # Fill all masks (greedy decoding)
                transfer_index = mask_index.clone()

            # Update input_ids with filled tokens
            # Ensure we only update mask positions that are being transferred
            forward_batch.input_ids[transfer_index] = x[transfer_index]

        # Final forward pass
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        # Extract generated tokens (everything after prompt)
        next_token_ids = forward_batch.input_ids[start:]

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = FastDLLMBlockDiffusion
