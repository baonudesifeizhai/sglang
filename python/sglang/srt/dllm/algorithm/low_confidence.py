from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


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
        # input_ids contains only the current block (mask tokens)
        # full_logits contains logits for the entire sequence (prefix + current block)
        # We need to extract only the logits for the current block

        num_input_tokens = len(forward_batch.input_ids)
        mask_index = forward_batch.input_ids == self.mask_id

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            # full_logits shape: [total_seq_len, vocab_size]
            # We only need the last num_input_tokens (the current block)
            block_logits = logits_output.full_logits[-num_input_tokens:]

            x = torch.argmax(block_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(block_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )
            x = torch.where(mask_index, x, forward_batch.input_ids)
            confidence = torch.where(mask_index, p, -np.inf)

            transfer_index = confidence > self.threshold
            if transfer_index.sum().item() == 0:
                _, select_index = torch.topk(confidence, k=1)
                transfer_index[select_index] = True

            forward_batch.input_ids[transfer_index] = x[transfer_index]

        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        # Return all tokens in the current block as the generated tokens
        next_token_ids = forward_batch.input_ids
        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = LowConfidence
