"""
PixArt-α Transformer model for image generation.

Adapted from PixArt-alpha official codebase.  Self-contained: no mmcv, xformers,
or timm dependencies.  Uses PyTorch scaled_dot_product_attention for all attention
ops (requires PyTorch ≥ 2.0).

Only the PixArt_XL_2 variant (depth=28, hidden=1152) is exposed.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable
from itertools import repeat


# ─────────────────────────── Utility helpers ────────────────────────────

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────── Basic building blocks ──────────────────────

class DropPath(nn.Module):
    """Drop path (stochastic depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Feed-forward MLP."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if isinstance(act_layer, nn.Module):
            self.act = act_layer
        elif isinstance(act_layer, type) or callable(act_layer):
            self.act = act_layer()
        else:
            raise TypeError(f"Unsupported act_layer: {type(act_layer)!r}")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """2-D image → patch embedding via Conv2d."""
    def __init__(self, img_size=256, patch_size=2, in_chans=4,
                 embed_dim=1152, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)                       # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        return x


# ─────────────────────────── Attention layers ───────────────────────────

class SelfAttention(nn.Module):
    """Multi-head self-attention (replaces WindowAttention / timm.Attention)."""
    def __init__(self, dim, num_heads=8, qkv_bias=True,
                 attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention (query = image tokens, key/value = caption)."""
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0.,
                 **block_kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        """
        x:    (B, N, C)   image tokens
        cond: (B, S, C)   caption tokens
        mask: (B, S)      attention mask  1=keep  0=pad   (or None)
        """
        B, N, C = x.shape
        q = (self.q_linear(x)
             .reshape(B, N, self.num_heads, self.head_dim)
             .transpose(1, 2))                                 # (B, H, N, D)
        kv = (self.kv_linear(cond)
              .reshape(B, -1, 2, self.num_heads, self.head_dim)
              .permute(2, 0, 3, 1, 4))                        # (2, B, H, S, D)
        k, v = kv[0], kv[1]

        attn_mask = None
        if mask is not None:
            # mask: (B, S) → (B, 1, 1, S) broadcast across heads & queries
            attn_mask = mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = torch.where(
                attn_mask > 0.5,
                torch.zeros_like(attn_mask),
                torch.full_like(attn_mask, float('-inf')),
            )

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ─────────────────────── Timestep / Caption embedders ───────────────────

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep → vector."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """Project caption embeddings to model dimension; stores the unconditional
    embedding for classifier-free guidance."""
    def __init__(self, in_channels, hidden_size, uncond_prob,
                 act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size,
                          out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer(
            "y_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class T2IFinalLayer(nn.Module):
    """Final layer with adaLN-single modulation."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5
        )
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─────────────────────────── PixArt Block ───────────────────────────────

class PixArtBlock(nn.Module):
    """A PixArt block with adaLN-single conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.,
                 input_size=None, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5
        )

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(
            gate_msa
            * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C)
        )
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(
            gate_mlp
            * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )
        return x


# ─────────────────────── Sinusoidal positional embedding ────────────────

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0,
                            lewei_scale=1.0, base_size=16):
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ─────────────────────────── PixArt Model ───────────────────────────────

class PixArt(nn.Module):
    """Diffusion Transformer backbone (PixArt-α).

    Parameters match the official PixArt-alpha implementation so that
    pre-trained checkpoints can be loaded directly.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path: float = 0.,
        caption_channels=4096,
        lewei_scale=1.0,
        model_max_length=120,
        **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = lewei_scale

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels,
                                     hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size

        # Fixed sin-cos positional embedding
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        dp = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                drop_path=dp[i],
                input_size=(input_size // patch_size, input_size // patch_size),
            )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    # ── forward passes ──────────────────────────────────────────────────

    def forward(self, x, timestep, y, mask=None, data_info=None, **kwargs):
        """
        x:        (B, C, H, W) latent
        timestep: (B,) diffusion timestep
        y:        (B, 1, L, C_cap) caption embeddings
        mask:     (B, L) attention mask  1=keep 0=pad   [optional]
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)

        self.h = x.shape[-2] // self.patch_size
        self.w = x.shape[-1] // self.patch_size

        x = self.x_embedder(x) + pos_embed          # (B, N, D)
        t = self.t_embedder(timestep.to(x.dtype))    # (B, D)
        t0 = self.t_block(t)                          # (B, 6D)

        y = self.y_embedder(y, self.training)         # (B, 1, L, D)
        y = y.squeeze(1)                              # (B, L, D)

        # Broadcast mask to match y batch dim (for CFG doubled batch)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            if mask.dim() > 2:
                mask = mask.view(mask.shape[0], -1)

        for block in self.blocks:
            x = block(x, y, t0, mask=mask)

        x = self.final_layer(x, t)                    # (B, N, p*p*C_out)
        x = self.unpatchify(x)                        # (B, C_out, H, W)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        """Return only the ε prediction (discard learned σ)."""
        model_out = self.forward(x, timestep, y, mask)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """Batched CFG forward (used by IDDPM sampler)."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # ── helpers ─────────────────────────────────────────────────────────

    def unpatchify(self, x):
        """(B, N, p²·C) → (B, C, H, W)"""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # sin-cos positional embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
            lewei_scale=self.lewei_scale,
            base_size=self.base_size,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # patch embedding like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # caption MLP
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # zero-out cross-attention projection
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # zero-out final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


# ─────────────────────────── Factory ────────────────────────────────────

def PixArt_XL_2(**kwargs):
    """PixArt-XL/2: depth 28 · hidden 1152 · patch 2 · heads 16."""
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
