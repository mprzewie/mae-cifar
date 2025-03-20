import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

try:
    from timm.layers.helpers import to_2tuple
    from timm.layers.mlp import Mlp
except ImportError:
    from timm.models.layers.mlp import Mlp
    from timm.models.layers.helpers import to_2tuple

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block, Attention, LayerScale
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


class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_reflections=1):
        super().__init__()
        assert in_features == out_features
        features = in_features
        self.features = features
        self.num_chunks = num_reflections

        # Householder vector (N-1 parameters)
        self.v = nn.Parameter(torch.randn(num_reflections, features - 1))

        # Rotation vectors for each chunk (num_chunks x (N-1))
        self.r = nn.Parameter(torch.randn(num_reflections, features - 1) * 0.1)
        # self.register_buffer("r", torch.zeros(features - 1))

        # Modulation vector (N parameters)
        self.m = nn.Parameter(torch.ones(features))


        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('bias', None)

    def construct_W(self):
        """Fully vectorized construction of K orthogonal matrices, outputting only the necessary rows."""
        N = self.features
        chunk_size = N // self.num_chunks  # Each chunk contributes this many rows
        device = self.r.device

        # Step 1: Construct Skew-Symmetric Rotation Matrices for Each Chunk
        R = torch.zeros((self.num_chunks, N, N), device=device)  # Shape: (K, N, N)

        # Correctly assign N-1 parameters per chunk to ensure skew-symmetry
        indices = torch.arange(1, N, device=device)

        # Fix: Now `self.r` is of shape (K, N-1) to allow independent rotations per chunk
        R[:, 0, indices] = self.r  # Rotate e2,...,eN around e1 for each chunk
        R[:, indices, 0] = -self.r  # Ensure skew-symmetry

        # Compute full batch of orthogonal rotation matrices
        Q_rotated = torch.matrix_exp(R)  # Shape: (K, N, N)

        # Step 2: Compute Householder Reflections in a Batch
        v_full = torch.cat([torch.ones(self.num_chunks, 1, device=device), self.v], dim=1)  # Shape: (K, N)
        v_full = v_full / v_full.norm(dim=1, keepdim=True)  # Normalize each vector

        H = torch.eye(N, device=device).expand(self.num_chunks, N, N) - \
            2 * v_full.unsqueeze(2) @ v_full.unsqueeze(1)  # Shape: (K, N, N)

        # Step 3: Apply Householder Reflection After Rotation
        Q = H @ Q_rotated  # Shape: (K, N, N)

        # Step 4: Select Only the First `chunk_size` Rows From Each Chunk
        Q_selected = Q[:, :chunk_size, :]  # Shape: (K, N//K, N)

        # Step 5: Concatenate Along the Correct Dimension to Form Final Weight Matrix
        W = torch.cat(torch.unbind(Q_selected, dim=0), dim=0)  # Shape: (N, N)

        # Step 6: Apply Modulation
        W = W * self.m.unsqueeze(1)  # Apply modulation

        return W

    def forward(self, x):
        W = self.construct_W()
        return torch.nn.functional.linear(x, W, self.bias)


class OrthoLinearContainer(nn.Module):
    def __init__(self, in_features, out_features, bias: bool=True, num_reflections: int=1):
        super().__init__()

        self.inner = nn.ModuleList()
        self.in_features = in_features
        self.out_features = out_features

        if out_features >= in_features:
            assert out_features % in_features == 0
            for _ in range(out_features // in_features):
                self.inner.append(OrthogonalLinear(in_features, in_features, bias=bias, num_reflections=num_reflections))
        else:
            assert in_features % out_features == 0
            for _ in range(in_features // out_features):
                self.inner.append(OrthogonalLinear(out_features, out_features, bias=bias, num_reflections=num_reflections))


    def forward(self, x):
        if self.out_features >= self.in_features:
            inner_out = [i(x) for i in self.inner]
            return torch.cat(inner_out, dim=-1)
        else:
            chunks = x.chunk(4, dim=-1)
            outs = torch.stack([i(c) for i, c in zip(self.inner, chunks)], dim=-1).sum(dim=-1)
            return outs
            # assert False, (x.shape, outs.shape)


class PatchShuffle(torch.nn.Module):
    def forward(self, patches: torch.Tensor, mask_ratio: float, forward_indexes = None, backward_indexes = None):
        T, B, C = patches.shape
        remain_T = int(T * (1 - mask_ratio))

        if forward_indexes is not None:
            assert backward_indexes is not None
        else:
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
            a : Attention = self.attn
            x = self.norm1(x)

            B, N, C = x.shape
            qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, a.proj.out_features // a.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)

            q = q * a.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = a.attn_drop(attn)
            return x_blk, attn

        return x_blk

class OrtoAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., orto_reflections: int = 0):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        if orto_reflections > 0:
            self.qkv = OrthoLinearContainer(dim, 3*dim, bias=qkv_bias, num_reflections=orto_reflections)
            self.proj = OrthogonalLinear(dim, dim, num_reflections=orto_reflections)


class OrtoMlp(Mlp):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., orto_reflections: int = 0):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop
        )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        if orto_reflections > 0:
            self.fc1 = OrthoLinearContainer(in_features, hidden_features, bias=bias[0], num_reflections=orto_reflections)
            self.fc2 = OrthoLinearContainer(hidden_features, out_features, bias=bias[1], num_reflections=orto_reflections)


class OrtoBlock(Block):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, orto_reflections: int = 0):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        if orto_reflections:
            self.attn = OrtoAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, orto_reflections=orto_reflections)
            self.mlp = OrtoMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, orto_reflections=orto_reflections)


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 orto_reflections: int = 0,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        # self.shuffle = PatchShuffle(mask_ratio)
        self.shuffle = PatchShuffle()
        # self.mask_ratio=mask_ratio

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[OrtoBlock(emb_dim, num_head, orto_reflections=orto_reflections) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(
            self, img, mask_ratio: float, return_attn_masks: bool =False, *,
            forward_indexes = None, backward_indexes = None,
            # latent_loss_block: int = 11,
    ):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches, mask_ratio=mask_ratio, forward_indexes=forward_indexes, backward_indexes=backward_indexes)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        x_ = patches


        # trans = self.transformer(patches)

        # latent_features = None
        # if return_attn_masks:
        # attns = []
        for bi, blk in enumerate(self.transformer):

            x_ = blk(x_)
            # attns.append(attn)
            # if bi == latent_loss_block:
            #     latent_features = x_

        # attns = torch.stack(attns, dim=1)

        trans = x_

        # assert latent_features is not None, f"{latent_loss_block=}, {len(self.transformer)=}"

        features = self.layer_norm(trans)
        features = rearrange(features, 'b t c -> t b c')
        # latent_features = rearrange(latent_features, 'b t c -> t b c')

        # if return_attn_masks:
        #     return features, latent_features, forward_indexes, backward_indexes, attns

        # return features, latent_features, forward_indexes, backward_indexes
        return features, forward_indexes, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 out_size: int = None,
                 orto_reflections: int = 0
                 ) -> None:
        super().__init__()
        out_size = out_size or 3 * patch_size ** 2
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[OrtoBlock(emb_dim, num_head, orto_reflections=orto_reflections) for _ in range(num_layer)])

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
                 latent_loss_block: int = 11,
                 latent_loss_detach_cls: bool = False,
                 orto_reflections: int = 0,
                 ) -> None:
        super().__init__()

        # self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.latent_loss_block = latent_loss_block

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, orto_reflections=orto_reflections)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, out_size=3 * patch_size ** 2, orto_reflections=orto_reflections)
        # self.l_decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, out_size=emb_dim)
        # self.l_decoder.patch2img = nn.Identity()

        self.mask_ratio_student = mask_ratio_student
        self.mask_ratio_teacher = mask_ratio_teacher
        self.latent_loss_detach_cls = latent_loss_detach_cls


    # def forward_l_decoder(self):
    def forward(self, img, *, forward_indexes = None, backward_indexes = None):
        features, forward_indexes, backward_indexes = self.encoder.forward(
            img, mask_ratio=self.mask_ratio_student,
            forward_indexes=forward_indexes, backward_indexes=backward_indexes,
            # latent_loss_block=self.latent_loss_block
        )

        # if self.mask_ratio_teacher >= 0:
        #     full_features, _, _, _ = self.encoder.forward(img, mask_ratio=self.mask_ratio_teacher)
        #     part_features= features
        #     full_cls_features = full_features[:1]
        #     mask_patch_features = part_features[1:]
        #     features = torch.cat([full_cls_features, mask_patch_features], dim=0)

        predicted_img, mask = self.decoder.forward(features,  backward_indexes)


        cls_features = features[:1]
        if self.latent_loss_detach_cls:
            cls_features = cls_features.detach()

        # mask_features = self.l_decoder.mask_token.expand(
        #     latent_features.shape[0]-1, latent_features.shape[1], -1
        # )

        ## predicting encoder features
        # l_features = torch.cat([cls_features, mask_features], dim=0)
        # l_pred, _ = self.l_decoder(l_features, backward_indexes)
        # l_pos_features = take_indexes(l_pred, forward_indexes)
        # l_pos_features = l_pos_features[:(l_features.shape[0] - 1)]

        # return predicted_img, mask, features, l_pos_features, (forward_indexes, backward_indexes)
        return predicted_img, mask, features, (forward_indexes, backward_indexes)

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10, linprobe:bool=False, num_last_blocks: int = 1) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm

        self.head = torch.nn.Linear(
            self.pos_embedding.shape[-1] * num_last_blocks, num_classes
        )
        self.linprobe = linprobe
        self.num_last_blocks = num_last_blocks

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        if self.num_last_blocks == 1:
            features = self.layer_norm(self.transformer(patches))
        else:
            outs = []
            x_ = patches

            for blk in self.transformer:
                x_ = blk(x_)
                outs.append(x_)

            features = torch.cat(outs[-self.num_last_blocks:], dim=2)


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
    # encoder = MAE_Encoder(orto_linear=True)
    encoder = MAE_Encoder(orto_reflections=False)
    # decoder = MAE_Decoder(orto_linear=True)
    decoder = MAE_Decoder(orto_reflections=False)
    features, fi, backward_indexes = encoder.forward(img, ratio)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)
