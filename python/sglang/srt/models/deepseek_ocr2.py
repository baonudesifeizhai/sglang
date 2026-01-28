import logging
from typing import Iterable, List, Optional, Set, Tuple, TypeAlias, Union

import torch
from torch import Tensor, nn

from sglang.srt.configs.deepseek_ocr2 import DeepseekOCR2Config
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepencoderv2 import (
    MlpProjector,
    build_qwen2_decoder_as_encoder,
    build_sam_vit_b,
)
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
from sglang.srt.models.transformers import maybe_prefix

NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]

logger = logging.getLogger(__name__)


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    if isinstance(embeddings, torch.Tensor):
        return embeddings.flatten(0, -2)
    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])
    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    if len(multimodal_embeddings) == 0:
        return inputs_embeds
    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype
    try:
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()
        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e
        raise ValueError("Error during masked scatter operation") from e
    return inputs_embeds


def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: int | list[int],
) -> torch.Tensor:
    if isinstance(placeholder_token_id, list):
        is_multimodal = torch.isin(
            input_ids,
            torch.tensor(placeholder_token_id, device=input_ids.device),
        )
    else:
        is_multimodal = input_ids == placeholder_token_id
    return _merge_multimodal_embeddings(
        inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


class DeepseekOCR2ForCausalLM(nn.Module):
    def __init__(
        self,
        *,
        config: DeepseekOCR2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # language model
        if self.text_config.topk_method == "noaux_tc":
            self.model = DeepseekV3ForCausalLM(
                config=self.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        elif not self.text_config.use_mla:
            self.model = DeepseekForCausalLM(
                config=self.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        else:
            self.model = DeepseekV2ForCausalLM(
                config=self.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )

        # vision modules
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_qwen2_decoder_as_encoder()
        self.projector = MlpProjector(
            {
                "projector_type": "linear",
                "input_dim": self.projector_config.input_dim,
                "n_embed": self.projector_config.n_embed,
            }
        )

        n_embed = self.projector_config.n_embed
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

        self.image_token_id = getattr(config, "image_token_id", None) or getattr(
            config, "image_token_index", None
        )

    def _parse_and_validate_image_input(self, **kwargs: object):
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)
        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None
        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of pixel values. Got type: {type(pixel_values)}"
            )
        if not isinstance(images_spatial_crop, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of image sizes. Got type: {type(images_spatial_crop)}"
            )
        if not isinstance(images_crop, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of image crop. Got type: {type(images_crop)}"
            )
        return [pixel_values, images_crop, images_spatial_crop]

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> list[torch.Tensor]:
        images_in_this_batch = []
        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                patches = images_crop[jdx][0].to(torch.bfloat16)
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]

                if torch.sum(patches).item() != 0:
                    local_features = self.vision_model(self.sam_model(patches))
                    local_features = self.projector(local_features)

                    global_features = self.vision_model(self.sam_model(image_ori))
                    global_features = self.projector(global_features)

                    local_features = local_features.view(-1, local_features.shape[-1])
                    global_features = global_features.view(
                        -1, global_features.shape[-1]
                    )

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [
                                global_features,
                                self.view_seperator[None, :],
                                local_features,
                            ],
                            dim=0,
                        )
                    else:
                        global_local_features = torch.cat(
                            [
                                local_features,
                                self.view_seperator[None, :],
                                global_features,
                            ],
                            dim=0,
                        )
                else:
                    global_features = self.vision_model(self.sam_model(image_ori))
                    global_features = self.projector(global_features)
                    global_features = global_features.view(
                        -1, global_features.shape[-1]
                    )
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(self, mm_items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.stack([item.feature for item in mm_items], dim=0).type(
            self.sam_model.dtype
        )
        images_crop = (
            torch.stack([item.images_crop for item in mm_items], dim=0)
            .type(torch.long)
            .to(device=pixel_values.device)
        )
        images_spatial_crop = (
            torch.cat([item.images_spatial_crop for item in mm_items], dim=0)
            .type(torch.long)
            .to(device=pixel_values.device)
        )
        assert images_crop.dim() == 6
        assert images_spatial_crop.dim() == 3

        vision_feature_lists = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )
        return torch.cat(vision_feature_lists, dim=0).type(self.sam_model.dtype)

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and self.image_token_id is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )
        return inputs_embeds

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return self._process_image_input(items)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name == "lm_head.weight":
                name = "model.lm_head.weight"
            elif name.startswith("model."):
                if any(
                    key in name
                    for key in [
                        "projector",
                        "vision_model",
                        "sam_model",
                        "view_seperator",
                    ]
                ):
                    name = name[len("model.") :]
                elif not (
                    ".projector" in name
                    or "vision_model" in name
                    or "sam_model" in name
                ):
                    name = name.replace("model.", "model.model.")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )


EntryClass = [DeepseekOCR2ForCausalLM]
