import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn

VIT_KWARGS = dict(
    vit_tiny=dict(
        emb_dim=192,
        encoder_layer=12,
        encoder_head=3,
        decoder_layer=4,
        decoder_head=3,
    ),
    vit_base=dict(
        emb_dim=768,
        encoder_layer=12,
        encoder_head=12,
        decoder_layer=8,
        decoder_head=16,
    )
)

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):

    def forward(self, patches : torch.Tensor, mask_ratio: float):
        T, B, C = patches.shape
        remain_T = int(T * (1 - mask_ratio))

        # remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class AttnBlock(Block):
    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        x_blk = super().forward(x)
        if return_attn:
            a = self.attn
            x = self.norm1(x)

            B, N, C = x.shape
            qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = a.q_norm(q), a.k_norm(k)
            q = q * a.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = a.attn_drop(attn)
            return x_blk, attn

        return x_blk

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        # self.shuffle = PatchShuffle(mask_ratio)
        self.shuffle = PatchShuffle()
        # self.mask_ratio=mask_ratio

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[AttnBlock(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, mask_ratio: float, return_attn_masks: bool =False):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches, mask_ratio=mask_ratio)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        x_ = patches
        trans = self.transformer(patches)

        if return_attn_masks:
            attns = []
            for blk in self.transformer:
                x_, attn = blk(x_, return_attn=True)
                attns.append(attn)

            attns = torch.stack(attns, dim=1)

            assert torch.allclose(x_, trans)
        # assert False, attns.shape


        features = self.layer_norm(trans)
        features = rearrange(features, 'b t c -> t b c')

        if return_attn_masks:
            return features, backward_indexes, attns

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 out_size: int = None
                 ) -> None:
        super().__init__()
        out_size = out_size or 3 * patch_size ** 2
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, out_size)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)
        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio_student=0.75,
                 mask_ratio_teacher=-1,
                 ) -> None:
        super().__init__()

        # self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, out_size=3 * patch_size ** 2)
        self.l_decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, out_size=emb_dim)
        self.l_decoder.patch2img = nn.Identity()

        self.mask_ratio_student = mask_ratio_student
        self.mask_ratio_teacher = mask_ratio_teacher


    # def forward_l_decoder(self):
    def forward(self, img):
        features, backward_indexes = self.encoder(img, mask_ratio=self.mask_ratio_student)

        if self.mask_ratio_teacher >= 0:
            full_features, _ = self.encoder(img, mask_ratio=self.mask_ratio_teacher)
            part_features= features
            full_cls_features = full_features[:1]
            mask_patch_features = part_features[1:]
            features = torch.cat([full_cls_features, mask_patch_features], dim=0)

        predicted_img, mask = self.decoder(features,  backward_indexes)


        ## predicting encoder features
        cls_features = features[:1]
        mask_features = self.l_decoder.mask_token.expand(features.shape[0]-1, features.shape[1], -1)
        l_features = torch.cat([cls_features, mask_features], dim=0)
        l_pred, _ = self.l_decoder(l_features, backward_indexes)
        forward_indices = torch.argsort(backward_indexes, dim=0)

        l_pos_features = take_indexes(l_pred, forward_indices)
        l_pos_features = l_pos_features[:(l_features.shape[0] - 1)]
        ###

        return predicted_img, mask, features, l_pos_features

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10, linprobe:bool=False) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)
        self.linprobe = linprobe
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        if self.linprobe:
            logits = self.head(features[0].detach())
        else:
            logits = self.head(features[0])
        return logits


if __name__ == '__main__':
    shuffle = PatchShuffle()
    ratio = 0.75
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a, ratio)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img, ratio)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
