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

This model is based on Qwen2 architecture. The dLLM-specific logic
(bidirectional attention, iterative decoding) is handled in the
dllm/algorithm/low_confidence.py, which directly calls the model
bypassing SGLang's KV cache machinery.
"""
from typing import Optional

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


class FastDLLMForCausalLM(Qwen2ForCausalLM):
    """
    Fast-dLLM v2 model.

    Inherits from Qwen2ForCausalLM. The dLLM-specific decoding logic
    is implemented in low_confidence.py which directly accesses the
    model's layers for bidirectional attention.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        # Override logits processor to return full logits
        # (needed for dLLM to get logits for all positions)
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)


# Use HuggingFace model name for registry
FastDLLMForCausalLM.__name__ = "Fast_dLLM_QwenForCausalLM"

EntryClass = FastDLLMForCausalLM
