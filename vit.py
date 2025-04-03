from functools import partial
from typing import List

import torch
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from timm.models.vision_transformer import _create_vision_transformer
from torch import nn
from model import OrtoBlock, APPLY_TO_ALL

ARCH_KWARGS={
    "vit_tiny_patch16": dict(patch_size=14, embed_dim=192, depth=12, num_heads=3),
    "vit_base_patch16": dict(patch_size=14, embed_dim=768, depth=12, num_heads=12)
}

def orto_vit(
    arch: str,
    input_size: List[int],
    num_classes: int,
    drop_rate: float,
    drop_path_rate: float,
    orto_reflections: int,
    orto_apply_to: str,
) -> VisionTransformer:
    arch_kwargs = ARCH_KWARGS[arch]

    c, h, w = input_size
    assert h == w, (h, w)

    if h // arch_kwargs["patch_size"] != 16:
        new_patch_size = h // 16
        print(f"{input_size=}, so adjusting patch size from {arch_kwargs['patch_size']} to {new_patch_size}")
        arch_kwargs["patch_size"] = new_patch_size

    block_fn = partial(
        OrtoBlock,
        orto_reflections=orto_reflections,
        apply_to=orto_apply_to,
    )

    return VisionTransformer(
        block_fn=block_fn,
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