from transformers import AutoProcessor, PretrainedConfig

from sglang.srt.configs.deepseek_ocr import (
    DeepseekOCRProcessor,
    DeepseekV2Config,
    MlpProjectorConfig,
    VisionEncoderConfig,
)
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


@register_customized_processor(processor_class=DeepseekOCRProcessor)
class DeepseekOCR2Config(PretrainedConfig):
    model_type = "deepseek-ocr2"
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: tuple[tuple[int, int]] = ((1024, 1024),)
    customized_processor_type = DeepseekOCRProcessor

    def __init__(
        self,
        tile_tag: str = "2D",
        global_view_pos: str = "head",
        candidate_resolutions: tuple[tuple[int, int]] = ((1024, 1024),),
        **kwargs,
    ):
        super().__init__(**kwargs)

        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get("projector_config", {})
        self.projector_config = MlpProjectorConfig(**projector_config)

        language_config = kwargs.get("language_config", {}) or kwargs.get(
            "text_config", {}
        )
        self.text_config = DeepseekV2Config(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size


AutoProcessor.register(DeepseekOCR2Config, DeepseekOCRProcessor)
