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
        # Debug info
        logger.info(f"input_ids shape: {forward_batch.input_ids.shape}")
        logger.info(f"input_ids first 10: {forward_batch.input_ids[:10].tolist()}")
        logger.info(f"mask_id: {self.mask_id}")

        mask_index = forward_batch.input_ids == self.mask_id
        num_masks = torch.sum(mask_index).item()
        logger.info(f"num_masks: {num_masks}")

        for iter_i in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                logger.info(f"iter {iter_i}: no more masks, breaking")
                break

            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            logger.info(
                f"iter {iter_i}: full_logits shape: {logits_output.full_logits.shape}"
            )

            x = torch.argmax(logits_output.full_logits, dim=-1)
            logger.info(
                f"iter {iter_i}: argmax x shape: {x.shape}, first 10: {x[:10].tolist()}"
            )

            p = torch.squeeze(
                torch.gather(
                    F.softmax(logits_output.full_logits, dim=-1),
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

            num_transferred = transfer_index.sum().item()
            logger.info(f"iter {iter_i}: transferred {num_transferred} tokens")

            forward_batch.input_ids[transfer_index] = x[transfer_index]

            if iter_i == 0:
                logger.info(
                    f"iter {iter_i}: input_ids after update: {forward_batch.input_ids[:10].tolist()}"
                )

        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        logger.info(f"final input_ids: {forward_batch.input_ids.tolist()}")
        next_token_ids = forward_batch.input_ids
        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = LowConfidence
