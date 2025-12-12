"""
LowConfidence decoding algorithm for diffusion LLM.

For dLLM, we bypass SGLang's KV cache mechanism and directly call the model
with the full sequence in each iteration. This is necessary because:
1. dLLM needs bidirectional attention (each token sees all other tokens)
2. The current block's tokens change during iteration (mask → generated token)
3. Each iteration needs to recompute attention for the current block
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LowConfidence(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        """
        Run LowConfidence decoding for dLLM.

        We directly call the model with the full sequence, bypassing SGLang's
        KV cache and attention backend machinery.
        """
        device = forward_batch.input_ids.device

        # Get the request
        assert forward_batch.batch_size == 1, "dLLM only supports batch_size=1"
        assert forward_batch.reqs is not None, "reqs must be provided for dLLM"
        req = forward_batch.reqs[0]

        # Get full sequence: prompt + current block (with masks)
        full_ids = torch.tensor(req.fill_ids, dtype=torch.int64, device=device)
        seq_len = len(full_ids)
        block_start = seq_len - self.block_size

        logger.info(
            f"=== LowConfidence START === seq_len={seq_len}, block_start={block_start}"
        )

        # Create position ids
        positions = torch.arange(seq_len, dtype=torch.int64, device=device)

        # Iteratively fill mask tokens
        for iter_i in range(self.block_size):
            # Check remaining masks in current block
            current_block = full_ids[block_start:]
            mask_index = current_block == self.mask_id
            remaining_masks = mask_index.sum().item()

            if remaining_masks == 0:
                break

            # Direct model forward (bypass SGLang's forward_batch machinery)
            with torch.no_grad():
                logits = self._direct_model_forward(model_runner, full_ids, positions)

            # Get logits for current block
            block_logits = logits[block_start:]

            # Compute predictions and confidence
            pred_tokens = torch.argmax(block_logits, dim=-1)
            probs = F.softmax(block_logits, dim=-1)
            confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

            # Only consider mask positions
            confidence = torch.where(
                mask_index, confidence, torch.tensor(-np.inf, device=device)
            )

            # Select tokens to transfer
            transfer_mask = confidence > self.threshold
            if not transfer_mask.any():
                # Transfer at least one token (highest confidence)
                best_idx = confidence.argmax()
                transfer_mask[best_idx] = True

            # Update full_ids with transferred tokens
            full_ids[block_start:] = torch.where(
                transfer_mask, pred_tokens, current_block
            )

            if iter_i < 3:
                n_transferred = transfer_mask.sum().item()
                logger.info(
                    f"iter {iter_i}: transferred {n_transferred}, remaining {remaining_masks - n_transferred}"
                )

        # Extract the generated block
        next_token_ids = full_ids[block_start:].clone()

        # Update req.fill_ids with generated tokens (for next block)
        req.fill_ids[block_start:] = next_token_ids.tolist()

        logger.info(
            f"=== LowConfidence END === generated: {next_token_ids[:10].tolist()}..."
        )

        # Return a dummy logits_output (we don't need it for dLLM)
        logits_output = LogitsProcessorOutput(
            next_token_logits=logits[-1:],  # Last token logits
            next_token_logprobs=None,
            normalized_prompt_logprobs=None,
            input_token_logprobs=None,
            output_top_logprobs=None,
            input_top_logprobs=None,
        )

        return logits_output, next_token_ids, False  # can_run_cuda_graph = False

    def _direct_model_forward(
        self,
        model_runner: ModelRunner,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Directly call the model without SGLang's ForwardBatch machinery.

        This bypasses KV cache and uses simple bidirectional attention.
        """
        model = model_runner.model

        # Get embeddings
        hidden_states = model.model.embed_tokens(input_ids)

        # Forward through transformer layers
        residual = None
        for layer in model.model.layers:
            if residual is None:
                residual = torch.zeros_like(hidden_states)

            hidden_states, residual = self._forward_layer(
                layer, positions, hidden_states, residual
            )

        # Final layer norm
        hidden_states = model.model.norm(hidden_states + residual)

        # Get logits by directly using lm_head weights
        # ParallelLMHead.forward() is disabled, so we compute logits manually
        logits = torch.matmul(hidden_states, model.lm_head.weight.t())
        if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None:
            logits = logits + model.lm_head.bias

        return logits

    def _forward_layer(
        self,
        layer,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through a single transformer layer with bidirectional attention.
        """
        # Input layernorm
        hidden_states = layer.input_layernorm(hidden_states + residual)

        # Self attention (we need to implement bidirectional attention)
        attn_output = self._bidirectional_attention(
            layer.self_attn, positions, hidden_states
        )

        # Residual connection
        residual = hidden_states
        hidden_states = attn_output

        # Post attention layernorm + MLP
        hidden_states = layer.post_attention_layernorm(hidden_states + residual)
        mlp_output = layer.mlp(hidden_states)

        residual = hidden_states
        hidden_states = mlp_output

        return hidden_states, residual

    def _bidirectional_attention(
        self,
        attn_module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bidirectional self-attention.
        """
        # Get Q, K, V
        qkv, _ = attn_module.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [attn_module.q_size, attn_module.kv_size, attn_module.kv_size], dim=-1
        )

        # Apply rotary embeddings
        q, k = attn_module.rotary_emb(positions, q, k)

        # Reshape for attention
        seq_len = hidden_states.shape[0]
        num_heads = attn_module.num_heads
        num_kv_heads = attn_module.num_kv_heads
        head_dim = attn_module.head_dim

        q = q.view(seq_len, num_heads, head_dim)
        k = k.view(seq_len, num_kv_heads, head_dim)
        v = v.view(seq_len, num_kv_heads, head_dim)

        # Expand KV heads if needed (GQA)
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Compute attention scores (bidirectional - no causal mask)
        # [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Scaled dot-product attention
        scale = 1.0 / (head_dim**0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(
            0, 1
        ).contiguous()  # [seq_len, num_heads, head_dim]
        attn_output = attn_output.view(seq_len, num_heads * head_dim)

        # Output projection
        output, _ = attn_module.o_proj(attn_output)

        return output


Algorithm = LowConfidence
