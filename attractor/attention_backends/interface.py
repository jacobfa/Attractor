"""Interface to attention backends - simplified with unified flash attention."""

import torch
from functools import partial
from typing import Callable, Optional

from .flash_attention import flash_attn


def attention_flash(q, k, v, mask=None, window_size=(-1, -1)):
    """
    Unified flash attention using FA3 on Hopper or SDPA fallback elsewhere.
    
    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        mask: Unused (for API compatibility), causal masking is always applied
        window_size: (left, right) sliding window. -1 means unlimited.
    
    Returns:
        Output tensor of shape (B, T, H, D)
    """
    return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)


def _skip_attention(q, k, v, mask=None):
    """For debugging/benchmarking without attention computation"""
    return v.clone()


def select_attention_implementation(
    provider="flash", window_size=(-1, -1), center=False, debias=False
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
    """
    Select attention implementation.
    
    Args:
        provider: "flash" (default, auto FA3/SDPA), "sdpa" (alias for flash), or "debug-skip"
        window_size: (left, right) sliding window for flash attention
        center: Deprecated, raises error if True
        debias: Deprecated, raises error if True
    
    Returns:
        Attention function with signature (q, k, v, mask) -> output
    """
    if center:
        raise ValueError("center attention is deprecated, use standard attention")
    if debias:
        raise ValueError("debias attention is deprecated, use standard attention")

    if provider in ("flash", "sdpa", "tridao"):
        return partial(attention_flash, window_size=window_size)
    elif provider == "debug-skip":
        return _skip_attention
    else:
        raise ValueError(f"Attention provider {provider!r} not supported. Use 'flash' or 'sdpa'.")
