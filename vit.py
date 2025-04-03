from functools import partial
from typing import List

import torch
from timm.models import VisionTransformer, PatchEmbed, Block
from timm.models.vision_transformer import _create_vision_transformer
from torch import nn
from model import OrtoBlock


def orto_vit(
    input_size: List[int],
    num_classes: int,
    drop_rate: float,
    drop_path_rate: float,
) -> VisionTransformer:
    arch_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return VisionTransformer(
        block_fn=...,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,

        **arch_kwargs,
    )

# def orto_vit_base_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, block_fn=...)
#     model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model