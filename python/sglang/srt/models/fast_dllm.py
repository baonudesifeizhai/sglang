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
# ==============================================================================

"""
Fast-dLLM v2 model for SGLang.

Based on Qwen2 architecture with modifications for diffusion LLM inference:
1. Uses bidirectional attention (controlled by is_dllm_model flag in flashinfer_backend)
2. Returns full logits for all tokens (required for dLLM inference)

Note: The bidirectional attention and KV cache behavior is handled by flashinfer_backend
when is_dllm_model=True, so we don't need ENCODER_ONLY attention type.
"""
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2 import Qwen2ForCausalLM, Qwen2Model
from sglang.srt.utils import add_prefix


class FastDLLMModel(Qwen2Model):
    """Model for Fast-dLLM, uses standard Qwen2 architecture."""

    pass


class FastDLLMForCausalLM(Qwen2ForCausalLM):
    """
    Fast-dLLM v2 model with full logits output for diffusion inference.

    Key differences from Qwen2ForCausalLM:
    1. Returns full logits for all tokens (return_full_logits=True)

    The bidirectional attention is handled by flashinfer_backend when
    is_dllm_model=True (detected via dllm_config).
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Call nn.Module.__init__ directly
        torch.nn.Module.__init__(self)

        from sglang.srt.distributed import get_pp_group
        from sglang.srt.layers.utils import PPMissingLayer
        from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # Use standard Qwen2Model - bidirectional attention is handled by backend
        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Handle lm_head
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()

        # Weight tying for PP
        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            elif self.pp_group.is_last_rank:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        # Return full logits for dLLM inference
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

        from sglang.srt.layers.pooler import Pooler, PoolingType

        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False


# Use HuggingFace model name for registry
FastDLLMForCausalLM.__name__ = "Fast_dLLM_QwenForCausalLM"

EntryClass = FastDLLMForCausalLM
