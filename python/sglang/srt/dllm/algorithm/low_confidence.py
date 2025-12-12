"""
LowConfidence decoding algorithm for diffusion LLM.

Based on Fast_dLLM's official implementation:
- Uses block-causal attention (not fully bidirectional)
- Each block can see previous blocks but not future blocks
- Iteratively fills mask tokens within each block
"""

import logging
from typing import Optional, Tuple, Union

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

        # fill_ids = origin_input_ids + [mask_id] * block_size (for current block)
        full_ids = torch.tensor(req.fill_ids, dtype=torch.int64, device=device)
        seq_len = len(full_ids)

        # Find the first mask token position - that's where current block starts
        mask_positions = (full_ids == self.mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) > 0:
            block_start = mask_positions[0].item()
        else:
            # No masks - shouldn't happen
            block_start = seq_len

        num_masks = len(mask_positions)

        logger.info(
            f"=== LowConfidence START === seq_len={seq_len}, block_start={block_start}, "
            f"num_masks={num_masks}"
        )

        # Create position ids for the CURRENT BLOCK only (not the full sequence)
        # This matches Fast_dLLM's behavior where positions are relative to block start
        positions = torch.arange(seq_len, dtype=torch.int64, device=device)

        # Iteratively fill mask tokens in current block
        for iter_i in range(self.block_size):
            # Check remaining masks in the current block
            mask_index = full_ids == self.mask_id
            remaining_masks = mask_index.sum().item()

            if remaining_masks == 0:
                break

            # Forward through model with block-causal attention
            with torch.no_grad():
                logits = self._forward_block_causal(
                    model_runner, full_ids, positions, self.block_size
                )

            # IMPORTANT: Shift logits like Fast_dLLM does
            # logits = torch.cat([logits[:1], logits[:-1]], dim=0)
            # Actually, for dLLM the logits at position i predict token at position i
            # So we need logits from the PREVIOUS position to predict current
            # But in SGLang, we directly use position i's logits for position i

            # Get predictions for mask positions only
            pred_tokens = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            confidence = probs.gather(-1, pred_tokens.unsqueeze(-1)).squeeze(-1)

            # Only consider mask positions for transfer
            confidence = torch.where(
                mask_index, confidence, torch.tensor(-float("inf"), device=device)
            )

            # Find tokens to unmask based on threshold
            unmask_idx = confidence > self.threshold

            # Always unmask at least one token (the highest confidence one)
            if unmask_idx.sum() == 0:
                max_conf_idx = confidence.argmax()
                unmask_idx[max_conf_idx] = True

            # Only unmask positions that are currently masked
            unmask_idx = unmask_idx & mask_index

            n_transferred = unmask_idx.sum().item()
            if iter_i < 3:
                transferred_tokens = pred_tokens[unmask_idx].tolist()
                transferred_conf = confidence[unmask_idx].tolist()
                logger.info(
                    f"iter {iter_i}: transferred {n_transferred}, remaining {remaining_masks - n_transferred}, "
                    f"tokens={transferred_tokens[:5]}, conf={[f'{c:.3f}' for c in transferred_conf[:5]]}"
                )

            # Update full_ids with predicted tokens
            full_ids[unmask_idx] = pred_tokens[unmask_idx]

        # Extract only the newly generated tokens (from block_start to end)
        next_token_ids = full_ids[block_start:]

        # Update req.fill_ids with the generated tokens
        for i, token_id in enumerate(next_token_ids.tolist()):
            req.fill_ids[block_start + i] = token_id

        logger.info(
            f"=== LowConfidence END === generated: {next_token_ids[:10].tolist()}..."
        )

        # Return logits output
        logits_output = LogitsProcessorOutput(
            next_token_logits=logits[-1:] if logits is not None else None,
            full_logits=logits,
        )

        return logits_output, next_token_ids, False

    def _forward_block_causal(
        self,
        model_runner: ModelRunner,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """
        Forward with block-causal attention mask.

        Block-causal means: tokens in block i can attend to all tokens in blocks <= i
        """
        model = model_runner.model
        seq_len = input_ids.shape[0]

        # Get embeddings
        hidden_states = model.model.embed_tokens(input_ids)

        # Create block-causal attention mask
        # block_q >= block_kv means current block can see previous blocks
        attn_mask = self._create_block_causal_mask(
            seq_len, block_size, input_ids.device
        )

        # Forward through transformer layers
        for layer in model.model.layers:
            hidden_states = self._forward_layer_with_mask(
                layer, positions, hidden_states, attn_mask
            )

        # Final layer norm
        hidden_states = model.model.norm(hidden_states)

        # Compute logits
        logits = torch.matmul(hidden_states, model.lm_head.weight.t())
        if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None:
            logits = logits + model.lm_head.bias

        return logits

    def _create_block_causal_mask(
        self, seq_len: int, block_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create block-causal attention mask.

        For position q_idx attending to position kv_idx:
        - block_q = q_idx // block_size
        - block_kv = kv_idx // block_size
        - Allow attention if block_q >= block_kv
        """
        q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)

        block_q = q_idx // block_size
        block_kv = kv_idx // block_size

        # True where attention is allowed
        mask = block_q >= block_kv

        # Convert to attention mask format (0 for allowed, -inf for blocked)
        attn_mask = torch.where(
            mask,
            torch.tensor(0.0, device=device),
            torch.tensor(-float("inf"), device=device),
        )

        return attn_mask

    def _forward_layer_with_mask(
        self,
        layer,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward through a decoder layer with custom attention mask.
        """
        seq_len = hidden_states.shape[0]

        # Input layernorm
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # Self attention with block-causal mask
        attn_output = self._attention_with_mask(
            layer.self_attn, positions, hidden_states, attn_mask
        )

        # Residual connection
        hidden_states = residual + attn_output

        # Post attention layernorm + MLP
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _attention_with_mask(
        self,
        attn_module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute self-attention with custom mask using PyTorch's SDPA.
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

        # Expand attention mask for batch and heads: [seq, seq] -> [1, 1, seq, seq]
        attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0)

        # Scaled dot-product attention with custom mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask_expanded, is_causal=False
        )

        # Reshape back: [1, num_heads, seq_len, head_dim] -> [seq_len, hidden]
        attn_output = (
            attn_output.squeeze(0).transpose(0, 1).contiguous().view(seq_len, -1)
        )

        # Output projection
        output, _ = attn_module.o_proj(attn_output)

        return output


Algorithm = LowConfidence
