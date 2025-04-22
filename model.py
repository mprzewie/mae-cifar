from email.policy import default
from typing import Optional

import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.vision_transformer import Mlp
try:
    from timm.layers.helpers import to_2tuple
    from timm.layers.mlp import Mlp
except ImportError:
    from timm.models.layers.mlp import Mlp
    from timm.models.layers.helpers import to_2tuple

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block, Attention, LayerScale
from torch import nn

APPLY_TO_ALL = "qkvpr1r2"
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
    def __init__(
            self, in_features: int, out_features: int, bias: bool=True,
            num_reflections=1, forward_impl: str="fast"
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        features = max(in_features, out_features)

        assert num_reflections == 1

        # Householder vector (N-1 parameters)
        self.v = nn.Parameter(torch.randn(features - 1))

        # Rotation vectors for each chunk (num_chunks x (N-1))
        self.r = nn.Parameter(torch.randn(features - 1) * 0.1)

        # Modulation vector (N parameters)
        self.m = nn.Parameter(torch.ones(out_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.forward_impl = forward_impl

    def construct_W(self):
        """Fully vectorized construction of K orthogonal matrices, outputting only the necessary rows."""
        N = max(self.in_features, self.out_features)
        device = self.r.device

        # Step 1: Construct Skew-Symmetric Rotation Matrices for Each Chunk
        R = torch.zeros((N, N), device=device)  # Shape: (K, N, N)

        # Correctly assign N-1 parameters per chunk to ensure skew-symmetry
        indices = torch.arange(1, N, device=device)

        R[0, indices] = self.r  # Rotate e2,...,eN around e1 for each chunk
        R[indices, 0] = -self.r  # Ensure skew-symmetry

        # # Compute full batch of orthogonal rotation matrices
        # Q_rotated = torch.matrix_exp(R)  # Shape: (K, N, N)

        # Rodrigues' Formula
        theta = self.r.norm() + 1e-8
        A = R / theta
        Q_rot2 = (
                torch.eye(R.shape[0], device=R.device)
                + torch.sin(theta) * A
                + (1 - torch.cos(theta)) * (A @ A)
        )
        Q_rotated = Q_rot2
        # assert torch.allclose(Q_rotated, Q_rot2, rtol=1e-4), (Q_rotated - Q_rot2).abs().max()

        # Step 2: Compute Householder Reflection (Fixing e1 -> v1)
        v_full = torch.cat([torch.tensor([1.0], device=device), self.v])  # Extend to full size
        v_full = v_full / v_full.norm()  # Normalize to be a unit vector
        H = torch.eye(N, device=device) - 2 * torch.outer(v_full, v_full)  # Householder matrix

        # Step 3: Apply Householder Reflection After Rotation
        W = H @ Q_rotated

        # Step 6: Cut out the relevant part of constructed W
        W = W[:self.out_features, :self.in_features]

        # Step 7: Apply Modulation
        W = W * self.m.unsqueeze(1)  # Apply modulation

        return W

    def forward(self, x):

        if self.forward_impl == "fast":
            return self.fast_forward(x)
        elif self.forward_impl == "slow":
            return self.slow_forward(x)
        elif self.forward_impl == "safe":
            out_slow = self.slow_forward(x)
            out_fast = self.fast_forward(x)
            assert torch.allclose(out_fast, out_slow, atol=1e-5)
            return out_fast

        assert False, "unknown forward implementation"

    @staticmethod
    def apply_ortho_operator(x, r, v, m):
        """
        Apply exp(R) x where R is skew-symmetric with nonzeros in first row/col defined by r.
        x: (B, N)
        r: (N - 1,)
        """
        device = x.device
        B = x.shape[0]
        I = x.shape[-1]
        O = len(m)
        N = max(I, O)
        # I = in_features, O = out_features, N = processing size

        if O > I:
            # if in size is lower, pad with zeros
            pad = torch.zeros(B, O - I, device=device)
            x = torch.cat((x, pad), dim=1)

        theta = r.norm() + 1e-8

        r_full = torch.cat([torch.tensor([1.0], device=device), r]).unsqueeze(0)  # Extend v with 1 for full vector
        r_unit = r_full / theta  # same normalization as A = R / θ

        x0 = x[:, :1]
        r_dot_x = torch.sum(x * r_unit, dim=1, keepdim=True)  # scalar per example

        e1 = torch.zeros(1, N, device=device)
        e1[0,0] = 1.0
        Ax_fast = r_dot_x * e1 - x0 * r_unit   # (B, N)

        Ax0 = Ax_fast[:, :1]
        r_dot_Ax = torch.sum(Ax_fast * r_unit, dim=1, keepdim=True)
        A2x_fast = r_dot_Ax * e1 - Ax0 * r_unit

        x_rot = x + torch.sin(theta) * Ax_fast + (1 - torch.cos(theta)) * A2x_fast

        # # Step 3: Apply Householder Reflection: Hx = x - 2vvᵀx / ‖v‖²
        v_full = torch.cat([torch.tensor([1.0], device=device), v])  # Extend v with 1 for full vector
        v_full = v_full / v_full.norm()  # Normalize v
        v_proj = torch.matmul(x_rot, v_full.unsqueeze(-1))  # Project x_rot onto v_full (B, N) * (N, 1) → (B, 1)
        v_proj = v_proj * v_full.unsqueeze(0)  # Broadcasting to (B, N)

        x_ref = x_rot - 2 * v_proj

        if O < I:
            # if out size is lower, cutout the latter part
            x_ref = x_ref[:, :O]

        # # Step 4: Apply modulation (scaling each vector by m)
        modulated = x_ref * m.unsqueeze(0)  # Apply modulation across the batch (B, N)

        return modulated

    def fast_forward(self, x):
        return self.apply_ortho_operator(x, self.r, self.v, self.m) + self.bias

    def slow_forward(self, x):
        W = self.construct_W()
        return torch.nn.functional.linear(x, W, self.bias)

    def _unittest_w_orthogonality(self, eps=1e-5):
        W = self.construct_W()
        # for i in range(W.shape[0]):
        #     for j in range(W.shape[1]):
        #         if i!= j:
        #             wi = W[i]
        #             wj = W[j]
        #             dot = (wi * wj).sum()
        #             assert torch.isclose(dot, torch.tensor(0.0, device=W.device), atol=eps), f"Rows {i}, {j} not orthogonal: dot={dot.item():.3e}"

        G = W @ W.T  # Gram matrix of row vectors
        G_diag = torch.diagonal(G)
        off_diag = G - torch.diag(G_diag)
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=eps), "W is not row-orthogonal" # Check if it's close to diagonal

        assert torch.allclose(G_diag, self.m ** 2, atol=eps), "Diagonal should be equal to m^2"

        row_norms = W.norm(dim=1)
        assert torch.allclose(row_norms, self.m.abs(), atol=eps), "Row norms not equal to m"
        print("W-orthogonal unittest OK")

    def _unittest_fast_forward(self, B=10, eps=1e-5):
        x = torch.randn(B, self.in_features).to(self.v.device)

        out_slow = self.forward(x)
        out_fast = self.fast_forward(x)

        assert torch.allclose(out_slow, out_fast, atol=eps), "Fast implementation has errors"
        print("Fast unittest OK")


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


class QKV(torch.nn.Module):
    def __init__(self, dim: int, bias: bool=True, num_reflections: int=1, apply_to: str = APPLY_TO_ALL):
        super().__init__()
        self.q = OrthogonalLinear(dim, dim, bias=bias, num_reflections=num_reflections)  if "q" in apply_to else nn.Linear(dim, dim, bias=bias)
        self.k = OrthogonalLinear(dim, dim, bias=bias, num_reflections=num_reflections)  if "k" in apply_to else nn.Linear(dim, dim, bias=bias)
        self.v = OrthogonalLinear(dim, dim, bias=bias, num_reflections=num_reflections)  if "v" in apply_to else nn.Linear(dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat([q,k,v], dim=-1)


class OrtoAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm: bool = False, attn_drop=0., proj_drop=0., norm_layer: nn.Module = nn.LayerNorm, orto_reflections: int = 0, apply_to: str = APPLY_TO_ALL):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        if orto_reflections > 0:
            self.qkv = QKV(dim, bias=qkv_bias, num_reflections=orto_reflections, apply_to=apply_to)
            if "p" in apply_to:
                self.proj = OrthogonalLinear(dim, dim, num_reflections=orto_reflections)


class OrtoMlp(Mlp):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., orto_reflections: int = 0, apply_to: str="r1r2"):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        if orto_reflections > 0:
            if "r1" in apply_to:
                self.fc1 = OrthogonalLinear(in_features, hidden_features, bias=bias[0], num_reflections=orto_reflections)
            if "r2" in apply_to:
                self.fc2 = OrthogonalLinear(hidden_features, out_features, bias=bias[1], num_reflections=orto_reflections)


class OrtoBlock(Block):
    def __init__(
            self,
            dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm: bool=False, proj_drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer: nn.Module = Mlp, orto_reflections: int = 0, apply_to: str = APPLY_TO_ALL):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer
        )
        self.attn = OrtoAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, orto_reflections=orto_reflections, apply_to=apply_to)
        if "r" in apply_to:
            self.mlp = OrtoMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop, orto_reflections=orto_reflections)


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 orto_reflections: int = 0,
                 force_linear_block_every: int = 1000000,
                 ortho_linear_apply_to: str = APPLY_TO_ALL
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        # self.shuffle = PatchShuffle(mask_ratio)
        self.shuffle = PatchShuffle()
        # self.mask_ratio=mask_ratio

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        blks = []
        for b in range(num_layer):
            b_orref = 0 if (b % force_linear_block_every == 0) else orto_reflections
            blks.append(
                OrtoBlock(emb_dim, num_head, orto_reflections=b_orref, apply_to=ortho_linear_apply_to)
            )
        self.transformer = torch.nn.Sequential(*blks)

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
                 orto_reflections: int = 0,
                 force_linear_block_every: int = 1000000,
                 ortho_linear_apply_to: str = APPLY_TO_ALL
                 ) -> None:
        super().__init__()
        out_size = out_size or 3 * patch_size ** 2
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        blks = []
        for b in range(num_layer):
            b_orref = 0 if (b % force_linear_block_every == 0) else orto_reflections
            blks.append(
                OrtoBlock(emb_dim, num_head, orto_reflections=b_orref, apply_to=ortho_linear_apply_to)
            )
        self.transformer = torch.nn.Sequential(*blks)

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
                 force_linear_block_every: int = 100000,
                 ortho_linear_apply_to: str=APPLY_TO_ALL
                 ) -> None:
        super().__init__()

        # self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.latent_loss_block = latent_loss_block

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, orto_reflections=orto_reflections, force_linear_block_every=force_linear_block_every, ortho_linear_apply_to=ortho_linear_apply_to)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, out_size=3 * patch_size ** 2, orto_reflections=orto_reflections, force_linear_block_every=force_linear_block_every, ortho_linear_apply_to=ortho_linear_apply_to)
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
    from time import time
    from collections import defaultdict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)

    OrthogonalLinear(512, 512).to(device)._unittest_w_orthogonality()
    OrthogonalLinear(512, 512).to(device)._unittest_fast_forward()


    results = dict(
        l11=defaultdict(list),
        l14=defaultdict(list),
        l41=defaultdict(list),
        # t11=defaultdict(list),
        # t14=defaultdict(list),
        # t41=defaultdict(list),
        o11=defaultdict(list),
        o11f=defaultdict(list),
        o11s=defaultdict(list),
        o14=defaultdict(list),
        o14f=defaultdict(list),
        o14s=defaultdict(list),
        o41f=defaultdict(list),
        o41s=defaultdict(list),
    )

    for emb in [512, 768, 1024, 2048]: #, 4096]:
        x = torch.randn((512, emb)).to(device)

        lrs = dict(
            l11=nn.Linear(emb, emb),
            l14 = nn.Linear(emb, 4 * emb),
            l41 = nn.Linear(emb, emb // 4),
            # t11=TorchLinear(emb, emb),
            # t14=TorchLinear(emb, 4 * emb),
            # t41=TorchLinear(emb, emb // 4),
            o11 = OrthogonalLinear(emb, emb, forward_impl="slow"),
            # o11 = OrthogonalLinear(emb, emb, forward_impl="fast"),
            o11f = OrthogonalLinear(emb, emb, forward_impl="fast"), #, mode="reduce-overhead", fullgraph=True),
            o14f = OrthogonalLinear(emb, 4 * emb, forward_impl="fast"),
            # o14s = OrthogonalLinear(emb, 4 * emb, forward_impl="safe"),
            o41f = OrthogonalLinear(emb, emb//4, forward_impl="fast"),
            # o41s = OrthogonalLinear(emb, emb//4, forward_impl="safe")
            # o11f=OrthogonalLinear(emb, emb, forward_impl="fast"),
            # o14f=OrthogonalLinear(emb, 4 * emb, forward_impl="fast"),
            # # o14s = OrthogonalLinear(emb, 4 * emb, forward_impl="safe"),
            # o41f=OrthogonalLinear(emb, emb // 4, forward_impl="fast"),
        )
        lrs = {k: v.to(device) for (k,v) in lrs.items()}

        from tqdm import tqdm
        for it in tqdm(range(100)):
            for l_name, l in lrs.items():
                s = time()
                y = l(x)
                t = time()
                # print(l_name)
                # if it > 500:
                results[l_name][emb].append(t-s)

        print(emb, {lr_name: np.mean(results[lr_name][emb]) for lr_name in lrs.keys()})

    import matplotlib.pyplot as plt
    for lr_name in list(results.keys()):
        X = sorted(results[lr_name].keys())
        Y = [np.mean(results[lr_name][emb]) for emb in X]
        std = [np.std(results[lr_name][emb]) for emb in X]
        ls = "-" if "l" in lr_name else "--" if "o" in lr_name else ":" #if "t" in lr_name
        plt.errorbar(X, Y, yerr=std, label=lr_name, linestyle=ls)

    plt.yscale("log")
    plt.legend()
    plt.show()


    # shuffle = PatchShuffle()
    # ratio = 0.75
    # a = torch.rand(16, 2, 10)
    # b, forward_indexes, backward_indexes = shuffle(a, ratio)
    # print(b.shape)
    #
    # img = torch.rand(2, 3, 32, 32)
    # # encoder = MAE_Encoder(orto_linear=True)
    # encoder = MAE_Encoder(orto_reflections=1)
    # opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    # # decoder = MAE_Decoder(orto_linear=True)
    # decoder = MAE_Decoder(orto_reflections=1)
    # features, fi, backward_indexes = encoder.forward(img, ratio)
    # print(forward_indexes.shape)
    # predicted_img, mask = decoder(features, backward_indexes)
    # print(predicted_img.shape)
    # loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    # loss.backward()
    # opt.step()
    # print(loss)
