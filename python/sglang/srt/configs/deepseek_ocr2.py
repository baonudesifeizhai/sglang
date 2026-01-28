import math
from typing import List

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, PretrainedConfig

from sglang.srt.configs.deepseek_ocr import (
    BASE_SIZE,
    DeepseekOCRProcessor,
    DeepseekV2Config,
    MlpProjectorConfig,
    VisionEncoderConfig,
    dynamic_preprocess,
)
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)

OCR2_LOCAL_IMAGE_SIZE = 768


class DeepseekOCR2Processor(DeepseekOCRProcessor):
    """OCR-2 processor with image tokenization aligned to OCR-2 embeddings."""

    def tokenize_with_images(
        self,
        conversation: str,
        images: List[Image.Image],
        bos: bool = True,
        eos: bool = True,
        cropping: bool = True,
    ):
        assert conversation.count(self.image_token) == len(images)
        text_splits = conversation.split(self.image_token)
        images_list, images_crop_list, images_seq_mask, images_spatial_crop = (
            [],
            [],
            [],
            [],
        )
        image_shapes = []
        num_image_tokens = []
        tokenized_str = []
        for text_sep, image in zip(text_splits, images):
            tokenized_sep = self.encode(text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)

            image_shapes.append(image.size)

            if (
                image.size[0] <= OCR2_LOCAL_IMAGE_SIZE
                and image.size[1] <= OCR2_LOCAL_IMAGE_SIZE
            ):
                crop_ratio = [1, 1]
            else:
                if cropping:
                    images_crop_raw, crop_ratio = dynamic_preprocess(
                        image, image_size=OCR2_LOCAL_IMAGE_SIZE
                    )
                else:
                    crop_ratio = [1, 1]

            if self.image_size <= OCR2_LOCAL_IMAGE_SIZE and not cropping:
                image = image.resize((self.image_size, self.image_size))

            global_view = ImageOps.pad(
                image,
                (BASE_SIZE, BASE_SIZE),
                color=tuple(int(x * 255) for x in self.image_transform.mean),
            )
            images_list.append(self.image_transform(global_view))

            num_width_tiles, num_height_tiles = crop_ratio
            images_spatial_crop.append([num_width_tiles, num_height_tiles])

            if num_width_tiles > 1 or num_height_tiles > 1:
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(self.image_transform(images_crop_raw[i]))

            num_queries = math.ceil(
                (OCR2_LOCAL_IMAGE_SIZE // self.patch_size) / self.downsample_ratio
            )
            num_queries_base = math.ceil(
                (BASE_SIZE // self.patch_size) / self.downsample_ratio
            )

            tokenized_image = [self.image_token_id] * (
                num_queries_base * num_queries_base
            )
            tokenized_image += [self.image_token_id]
            if num_width_tiles > 1 or num_height_tiles > 1:
                tokenized_image += (
                    [self.image_token_id]
                    * (num_queries * num_width_tiles)
                    * (num_queries * num_height_tiles)
                )
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            num_image_tokens.append(len(tokenized_image))

        tokenized_sep = self.encode(text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if bos:
            tokenized_str = [self.bos_id] + tokenized_str
            images_seq_mask = [False] + images_seq_mask
        if eos:
            tokenized_str = tokenized_str + [self.eos_id]
            images_seq_mask = images_seq_mask + [False]

        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)

        input_ids = torch.LongTensor(tokenized_str)
        target_ids = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        target_ids[(input_ids < 0) | (input_ids == self.image_token_id)] = (
            self.ignore_id
        )
        input_ids[input_ids < 0] = self.pad_id

        if eos:
            assert input_ids[-1] == self.eos_id
            input_ids = input_ids[:-1]
            target_ids = target_ids[:-1]
            images_seq_mask = images_seq_mask[:-1]

        if len(images_list) == 0:
            pixel_values = torch.zeros((1, 3, BASE_SIZE, BASE_SIZE))
            images_spatial_crop = torch.zeros((1, 1), dtype=torch.long)
            images_crop = torch.zeros(
                (1, 3, OCR2_LOCAL_IMAGE_SIZE, OCR2_LOCAL_IMAGE_SIZE)
            ).unsqueeze(0)
        else:
            pixel_values = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0).unsqueeze(0)
            else:
                images_crop = torch.zeros(
                    (1, 3, OCR2_LOCAL_IMAGE_SIZE, OCR2_LOCAL_IMAGE_SIZE)
                ).unsqueeze(0)

        input_ids = input_ids.unsqueeze(0)
        return (
            input_ids,
            pixel_values,
            images_crop,
            images_seq_mask,
            images_spatial_crop,
            num_image_tokens,
            image_shapes,
        )


@register_customized_processor(processor_class=DeepseekOCR2Processor)
class DeepseekOCR2Config(PretrainedConfig):
    model_type = "deepseek-ocr2"
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: tuple[tuple[int, int]] = ((1024, 1024),)
    customized_processor_type = DeepseekOCR2Processor

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


AutoProcessor.register(DeepseekOCR2Config, DeepseekOCR2Processor)
