import math
import torch
from einops import rearrange
from torch import Tensor


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, return_attn_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """
    Compute attention with optional attention weight return.
    
    Args:
        q, k, v: Query, Key, Value tensors [B, H, L, D]
        pe: Positional encoding
        return_attn_weights: If True, also return attention weights [B, H, L, L]
    
    Returns:
        x: Attention output [B, L, H*D]
        attn_weights (optional): Attention weights [B, H, L, L]
    """
    q, k = apply_rope(q, k, pe)

    if return_attn_weights:
        scale = 1.0 / math.sqrt(q.size(-1))
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        x = torch.matmul(attn_weights, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x, attn_weights
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
