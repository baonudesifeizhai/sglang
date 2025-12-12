"""
LowConfidence decoding algorithm for diffusion LLM.

This algorithm iteratively fills mask tokens based on confidence scores.
For dLLM, we need to process the full sequence (prompt + current block)
in each iteration, not just the mask tokens.
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

    def __init__(
        self,
        config: DllmConfig,
    ):
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

        The key insight is that dLLM needs to see the full sequence (prompt + current block)
        in each iteration. We achieve this by:
        1. Getting the full sequence from req.fill_ids
        2. Creating a new input_ids tensor with the full sequence
        3. Iteratively filling mask tokens based on confidence
        4. Only returning the newly generated tokens (the current block)
        """
        device = forward_batch.input_ids.device

        # Get the request - we only support batch_size=1 for dLLM
        assert forward_batch.batch_size == 1, "dLLM only supports batch_size=1"
        assert forward_batch.reqs is not None, "reqs must be provided for dLLM"
        req = forward_batch.reqs[0]

        # Get the full sequence (prompt + all previous blocks + current block masks)
        full_ids = torch.tensor(req.fill_ids, dtype=torch.int64, device=device)
        seq_len = len(full_ids)

        # Find the start position of current block's mask tokens
        # This is where the new tokens will be generated
        block_start = seq_len - self.block_size

        logger.info(f"=== LowConfidence.run START ===")
        logger.info(f"full_ids length: {seq_len}, block_start: {block_start}")
        logger.info(f"full_ids last 40: {full_ids[-40:].tolist()}")

        # Create positions for the full sequence
        positions = torch.arange(seq_len, dtype=torch.int32, device=device)

        # Iteratively fill mask tokens
        for iter_i in range(self.block_size):
            # Find mask positions in the current block
            current_block = full_ids[block_start:]
            mask_index_block = current_block == self.mask_id
            remaining_masks = torch.sum(mask_index_block).item()

            if remaining_masks == 0:
                logger.info(f"iter {iter_i}: no more masks, breaking")
                break

            # Create a temporary ForwardBatch with full sequence
            # We need to bypass KV cache and compute attention over the full sequence
            temp_batch = self._create_full_seq_batch(
                forward_batch, full_ids, positions, model_runner
            )

            logits_output, can_run_cuda_graph = model_runner.forward(
                temp_batch, pp_proxy_tensors=None
            )

            if iter_i == 0:
                logger.info(
                    f"iter {iter_i}: full_logits shape: {logits_output.full_logits.shape}"
                )

            # Get logits for the current block only
            block_logits = logits_output.full_logits[block_start:]

            # Compute predictions and confidence
            x = torch.argmax(block_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(block_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            # Only consider mask positions for transfer
            confidence = torch.where(
                mask_index_block, p, torch.tensor(-np.inf, device=device)
            )

            # Select tokens to transfer based on confidence threshold
            transfer_index = confidence > self.threshold
            if transfer_index.sum().item() == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True

            num_transferred = transfer_index.sum().item()
            if iter_i < 3:
                transferred_positions = torch.where(transfer_index)[0].tolist()
                transferred_tokens = x[transfer_index].tolist()
                logger.info(
                    f"iter {iter_i}: transferred {num_transferred} tokens at block positions {transferred_positions}, tokens: {transferred_tokens}"
                )

            # Update full_ids with transferred tokens
            full_ids[block_start:][transfer_index] = x[transfer_index]

        # Final forward pass to get logits
        temp_batch = self._create_full_seq_batch(
            forward_batch, full_ids, positions, model_runner
        )
        logits_output, can_run_cuda_graph = model_runner.forward(
            temp_batch, pp_proxy_tensors=None
        )

        # Return only the current block's tokens
        next_token_ids = full_ids[block_start:]

        logger.info(f"final block tokens: {next_token_ids.tolist()}")
        logger.info(f"=== LowConfidence.run END ===")

        return logits_output, next_token_ids, can_run_cuda_graph

    def _create_full_seq_batch(
        self,
        original_batch: ForwardBatch,
        full_ids: torch.Tensor,
        positions: torch.Tensor,
        model_runner: ModelRunner,
    ) -> ForwardBatch:
        """
        Create a ForwardBatch for the full sequence.

        Key changes:
        - input_ids: full sequence (prompt + current block)
        - positions: 0 to seq_len-1
        - extend_prefix_lens_cpu: [0] to disable KV cache reading
        - extend_seq_lens_cpu: [seq_len]
        """
        seq_len = len(full_ids)
        device = full_ids.device

        # Create a new batch with full sequence
        # We set extend_prefix_lens to 0 to disable KV cache reading
        # This forces the model to compute attention over the full sequence
        new_batch = ForwardBatch(
            forward_mode=original_batch.forward_mode,
            batch_size=1,
            input_ids=full_ids,
            req_pool_indices=original_batch.req_pool_indices,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            out_cache_loc=original_batch.out_cache_loc,
            seq_lens_sum=seq_len,
            positions=positions,
            extend_num_tokens=seq_len,
            extend_seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32, device=device),
            extend_start_loc=torch.tensor([0], dtype=torch.int32, device=device),
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[seq_len],
            req_to_token_pool=original_batch.req_to_token_pool,
            token_to_kv_pool=original_batch.token_to_kv_pool,
            attn_backend=original_batch.attn_backend,
            sampling_info=original_batch.sampling_info,
        )

        return new_batch


Algorithm = LowConfidence
