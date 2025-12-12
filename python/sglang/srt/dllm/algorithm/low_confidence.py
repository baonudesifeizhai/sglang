"""
LowConfidence decoding algorithm for diffusion LLM.

For dLLM, we need bidirectional attention and iterative mask filling.
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
        device = forward_batch.input_ids.device

        # Get the request
        assert forward_batch.batch_size == 1, "dLLM only supports batch_size=1"
        assert forward_batch.reqs is not None, "reqs must be provided for dLLM"
        req = forward_batch.reqs[0]

        # fill_ids = origin_input_ids + [mask_id] * block_size
        # We need to find where the masks start
        full_ids = torch.tensor(req.fill_ids, dtype=torch.int64, device=device)
        seq_len = len(full_ids)

        # Find the first mask token position - that's where generated content starts
        mask_positions = (full_ids == self.mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) > 0:
            output_start = mask_positions[0].item()
        else:
            # No masks - shouldn't happen, but handle gracefully
            output_start = seq_len

        logger.info(
            f"=== LowConfidence START === seq_len={seq_len}, output_start={output_start}, "
            f"num_masks={len(mask_positions)}"
        )

        # Create position ids
        positions = torch.arange(seq_len, dtype=torch.int64, device=device)

        # Iteratively fill mask tokens
        for iter_i in range(self.block_size):
            # Check remaining masks
            mask_index = full_ids == self.mask_id
            remaining_masks = mask_index.sum().item()

            if remaining_masks == 0:
                break

            # Forward through model with bidirectional attention
            with torch.no_grad():
                logits = self._forward_bidirectional(model_runner, full_ids, positions)

            # Compute predictions and confidence for ALL positions
            pred_tokens = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

            # Only consider mask positions for transfer
            confidence = torch.where(
                mask_index, confidence, torch.tensor(-np.inf, device=device)
            )

            # Select tokens to transfer
            transfer_mask = confidence > self.threshold
            if not transfer_mask.any():
                # Transfer at least one token (highest confidence)
                best_idx = confidence.argmax()
                transfer_mask = torch.zeros_like(mask_index)
                transfer_mask[best_idx] = True

            # Update full_ids with transferred tokens
            full_ids = torch.where(transfer_mask, pred_tokens, full_ids)

            if iter_i < 3:
                n_transferred = transfer_mask.sum().item()
                logger.info(
                    f"iter {iter_i}: transferred {n_transferred}, remaining {remaining_masks - n_transferred}"
                )

        # Extract only the generated tokens (from output_start to end)
        next_token_ids = full_ids[output_start:]

        # Update req.fill_ids with generated tokens
        req.fill_ids = full_ids.tolist()

        logger.info(
            f"=== LowConfidence END === generated: {next_token_ids[:10].tolist()}..."
        )

        # Return logits output
        logits_output = LogitsProcessorOutput(
            next_token_logits=logits[-1:] if logits is not None else None,
            full_logits=logits,
        )

        return logits_output, next_token_ids, False

    def _forward_bidirectional(
        self,
        model_runner: ModelRunner,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward through model with bidirectional attention.
        Uses PyTorch's scaled_dot_product_attention for simplicity.
        """
        model = model_runner.model
        seq_len = len(input_ids)

        # Get embeddings [seq_len, hidden_size]
        hidden_states = model.model.embed_tokens(input_ids)

        # Forward through transformer layers
        residual = None
        for layer in model.model.layers:
            hidden_states, residual = self._forward_layer_bidirectional(
                layer, positions, hidden_states, residual
            )

        # Final layer norm
        hidden_states, _ = model.model.norm(hidden_states, residual)

        # Compute logits using lm_head weights directly
        logits = F.linear(hidden_states, model.lm_head.weight)

        return logits

    def _forward_layer_bidirectional(
        self,
        layer,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through a decoder layer with bidirectional attention.
        """
        seq_len = hidden_states.shape[0]

        # Input layernorm (fused with residual)
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)

        # Self attention with bidirectional mask
        attn_output = self._bidirectional_attention(
            layer.self_attn, positions, hidden_states
        )

        # Post attention layernorm (fused with residual)
        hidden_states, residual = layer.post_attention_layernorm(attn_output, residual)

        # MLP
        hidden_states = layer.mlp(hidden_states)

        return hidden_states, residual

    def _bidirectional_attention(
        self,
        attn_module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bidirectional self-attention using PyTorch's SDPA.
        """
        seq_len = hidden_states.shape[0]

        # Get Q, K, V
        qkv, _ = attn_module.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [attn_module.q_size, attn_module.kv_size, attn_module.kv_size], dim=-1
        )

        # Apply rotary embeddings
        q, k = attn_module.rotary_emb(positions, q, k)

        # Reshape for attention: [seq_len, hidden] -> [1, num_heads, seq_len, head_dim]
        num_heads = attn_module.num_heads
        num_kv_heads = attn_module.num_kv_heads
        head_dim = attn_module.head_dim

        q = q.view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
        k = k.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
        v = v.view(seq_len, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

        # Expand KV heads for GQA
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention (bidirectional - no causal mask)
        # Shape: [1, num_heads, seq_len, head_dim]
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False
        )

        # Reshape back: [1, num_heads, seq_len, head_dim] -> [seq_len, hidden]
        attn_output = (
            attn_output.squeeze(0).transpose(0, 1).contiguous().view(seq_len, -1)
        )

        # Output projection
        output, _ = attn_module.o_proj(attn_output)

        return output


Algorithm = LowConfidence
