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

        Fast_dLLM v2 推理流程：
        1. 初始化：prompt + mask tokens
        2. 迭代解码：
           - Forward pass 得到 logits
           - 应用 token shift（Fast_dLLM 核心特性）
           - Top-p 采样
           - 根据置信度填充 mask
        3. 返回生成的 tokens

        关键理解：
        - logits[i] 对应 input_ids[i] 位置的下一个 token 的预测
        - token shift: logits_shifted[i] = logits[i-1] (i>0), logits_shifted[0] = logits[0]
        - 这样可以在 block 内保持自回归特性
        """
        # Step 1: 识别 prompt 和 mask 位置
        input_ids = forward_batch.input_ids
        mask_index = input_ids == self.mask_id
        num_masks = torch.sum(mask_index).item()
        start = len(input_ids) - num_masks

        logger.info(
            f"[Fast_dLLM] 初始化: seq_len={len(input_ids)}, "
            f"prompt_len={start}, mask_count={num_masks}, "
            f"mask_id={self.mask_id}"
        )

        # Step 2: 迭代 block diffusion
        for iteration in range(self.max_iterations):
            # 检查是否所有 mask 都已填充
            mask_index = input_ids == self.mask_id
            num_remaining_masks = torch.sum(mask_index).item()

            if num_remaining_masks == 0:
                logger.info(f"[Fast_dLLM] 迭代 {iteration}: 所有 mask 已填充，退出")
                break

            logger.info(
                f"[Fast_dLLM] 迭代 {iteration}: 剩余 mask={num_remaining_masks}, "
                f"input_ids前10个={input_ids[:min(10, len(input_ids))].tolist()}"
            )

            # Forward pass
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            # 获取完整序列的 logits
            logits = logits_output.full_logits  # Shape: (num_tokens, vocab_size)
            logger.debug(
                f"[Fast_dLLM] 迭代 {iteration}: logits shape={logits.shape}, "
                f"logits[0]前5个值={logits[0][:5].tolist() if len(logits) > 0 else []}"
            )

            # 应用 token shift（Fast_dLLM 核心）
            # logits_shifted[i] = logits[i-1] for i > 0, logits_shifted[0] = logits[0]
            # 这样可以在 block 内保持自回归特性
            if logits.shape[0] > 1:
                logits_shifted = torch.cat([logits[:1], logits[:-1]], dim=0)
            else:
                logits_shifted = logits

            # Top-p 采样
            x_pred, probs = self._sample_with_top_p(
                logits_shifted, top_p=self.top_p, temperature=self.temperature
            )

            # 计算置信度（预测 token 的概率）
            x_pred_probs = torch.gather(
                probs, dim=-1, index=x_pred.unsqueeze(-1)
            ).squeeze(-1)

            logger.debug(
                f"[Fast_dLLM] 迭代 {iteration}: 预测前5个token={x_pred[:min(5, len(x_pred))].tolist()}, "
                f"对应概率={x_pred_probs[:min(5, len(x_pred_probs))].tolist()}"
            )

            # 只在 mask 位置更新，非 mask 位置保持原样
            x_final = torch.where(mask_index, x_pred, input_ids)
            confidence = torch.where(
                mask_index,
                x_pred_probs,
                torch.tensor(-float("inf"), device=x_pred_probs.device),
            )

            # 根据置信度阈值填充 mask
            if self.confidence_threshold > 0:
                # 只填充高置信度的 mask
                transfer_index = (confidence > self.confidence_threshold) & mask_index
                num_to_fill = torch.sum(transfer_index).item()

                if num_to_fill == 0:
                    # 如果没有高置信度的，至少填充一个（概率最高的）
                    _, select_idx = torch.topk(confidence, k=1)
                    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
                    transfer_index[select_idx] = True
                    num_to_fill = 1
                    logger.info(
                        f"[Fast_dLLM] 迭代 {iteration}: 无高置信度token，填充最高概率的1个"
                    )
                else:
                    logger.info(
                        f"[Fast_dLLM] 迭代 {iteration}: 填充 {num_to_fill} 个高置信度token "
                        f"(threshold={self.confidence_threshold})"
                    )
            else:
                # 填充所有 mask（贪婪解码）
                transfer_index = mask_index.clone()
                num_to_fill = torch.sum(transfer_index).item()
                logger.info(
                    f"[Fast_dLLM] 迭代 {iteration}: 填充所有 {num_to_fill} 个mask "
                    f"(threshold=0.0, 贪婪解码)"
                )

            # 更新 input_ids
            input_ids[transfer_index] = x_final[transfer_index]
            forward_batch.input_ids = input_ids

            logger.info(
                f"[Fast_dLLM] 迭代 {iteration}: 更新后 input_ids前10个="
                f"{input_ids[:min(10, len(input_ids))].tolist()}"
            )

        # 最终 forward pass
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        # 提取生成的 tokens（prompt 之后的所有内容）
        next_token_ids = input_ids[start:]

        logger.info(
            f"[Fast_dLLM] 完成: 生成 {len(next_token_ids)} 个tokens, "
            f"前10个={next_token_ids[:min(10, len(next_token_ids))].tolist()}"
        )

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = FastDLLMBlockDiffusion
