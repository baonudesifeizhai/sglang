from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        hf_config = model_config.hf_config
        arch = hf_config.architectures[0]

        # Get block_size and mask_id based on architecture
        if arch == "LLaDA2MoeModelLM":
            block_size = 32
            mask_id = 156895
        elif arch == "FastDLLMForCausalLM":
            block_size = getattr(hf_config, "bd_size", 32)
            mask_id = getattr(hf_config, "mask_token_id", 151665)
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        # Load algorithm config from YAML if provided
        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            import yaml

            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
        )
