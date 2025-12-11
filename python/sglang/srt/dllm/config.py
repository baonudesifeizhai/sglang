from typing import Any, Dict, List, Optional

import yaml

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


def _get_model_specific_config(
    architecture: str, server_args: ServerArgs
) -> tuple[int, int]:
    """
    Get model-specific configuration (mask_id and block_size).

    Returns:
        tuple: (mask_id, block_size)
    """
    if architecture == "LLaDA2MoeModelLM":
        mask_id = 156895
        block_size = server_args.dllm_block_size or 32
        return mask_id, block_size
    else:
        raise RuntimeError(f"Unknown diffusion LLM: {architecture}")


def _load_algorithm_config(config_path: str) -> Dict[str, Any]:
    """
    Load algorithm configuration from YAML file.

    Similar to load_pdmux_config() pattern in sglang/srt/multiplex/pdmux_context.py

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed config data dictionary (empty dict if file is empty or None)
    """
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    return config_data if config_data else {}


def _parse_step_map(config_data: Dict[str, Any]) -> Optional[Dict[int, List[int]]]:
    """
    Parse step_map from config data.

    Args:
        config_data: Parsed YAML config data

    Returns:
        step_map dictionary or None
    """
    if "step_map" not in config_data or not config_data["step_map"]:
        return None

    step_map_raw = config_data["step_map"]
    return {
        int(k): [int(x) for x in v]
        for k, v in step_map_raw.items()
    }


def _parse_algorithm_config_file(
    config_path: str,
) -> tuple[Optional[Dict[int, List[int]]], Dict[str, Any], Optional[int]]:
    """
    Parse algorithm configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        tuple: (step_map, algorithm_config, block_size_override)
        - step_map: Step map dictionary or None
        - algorithm_config: Other algorithm configs (excluding step_map and block_size)
        - block_size_override: block_size value if present in config, None otherwise
    """
    config_data = _load_algorithm_config(config_path)

    step_map = _parse_step_map(config_data)

    # Extract block_size if present
    block_size_override = config_data.get("block_size")

    # Extract other configs (excluding step_map and block_size)
    algorithm_config = {
        k: v for k, v in config_data.items()
        if k not in ("step_map", "block_size")
    }

    return step_map, algorithm_config, block_size_override


class DllmConfig:
    def __init__(
        self,
        mask_id: int,
        block_size: int,
        algorithm: str,
        step_map: Optional[Dict[int, List[int]]] = None,
        algorithm_config: Optional[Dict[str, Any]] = None,
    ):
        self.algorithm = algorithm
        self.block_size = block_size
        self.mask_id = mask_id
        self.step_map = step_map
        self.algorithm_config = algorithm_config or {}

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
        mask_id, block_size = _get_model_specific_config(architecture, server_args)

        # Parse step_map and algorithm_config from YAML file if provided
        step_map = None
        algorithm_config = {}

        if (
            hasattr(server_args, "dllm_algorithm_config")
            and server_args.dllm_algorithm_config
        ):
            step_map, algorithm_config, block_size_override = _parse_algorithm_config_file(
                server_args.dllm_algorithm_config
            )
            if block_size_override is not None:
                block_size = block_size_override

        return DllmConfig(
            mask_id=mask_id,
            block_size=block_size,
            algorithm=server_args.dllm_algorithm,
            step_map=step_map,
            algorithm_config=algorithm_config,
        )
