from typing import Any, Tuple

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


def get_architecture_config(architecture: str, hf_config: Any) -> Tuple[int, int]:
    if architecture == "LLaDA2MoeModelLM":
        block_size = 32
        mask_id = 156895
    elif architecture == "Fast_dLLM_QwenForCausalLM":
        block_size = getattr(hf_config, "bd_size", 32)
        mask_id = 151665
    else:
        raise RuntimeError(
            f"Unknown diffusion LLM: {architecture}. "
            f"Supported architectures: LLaDA2MoeModelLM, Fast_dLLM_QwenForCausalLM"
        )
    return block_size, mask_id


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
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        architecture = model_config.hf_config.architectures[0]
        block_size, mask_id = get_architecture_config(
            architecture, model_config.hf_config
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
        )
