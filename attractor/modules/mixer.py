import math
import torch
from torch import Tensor
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from attractor.models.config import AnyConfig

from attractor.modules.utils import apply_rotary_emb_complex_like
from attractor.attention_backends import flash_attn


def norm(x: Tensor) -> Tensor:
    """Purely functional rmsnorm with no learnable params."""
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: "AnyConfig", layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.head_dim = config.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_head

        assert config.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = config.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias, init_method=config.init.fn("q", layer_id))
        self.c_k = config.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias, init_method=config.init.fn("k", layer_id))
        self.c_v = config.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias, init_method=config.init.fn("v", layer_id))
        self.c_proj = config.Linear(config.n_embd, config.n_embd, bias=config.bias, init_method=config.init.fn("out_attn", layer_id))

        if config.qk_bias:
            self.qk_bias = torch.nn.Parameter(torch.zeros(2, 1, self.n_head, self.head_dim))
        self.ve_gate_channels = 32
        self.ve_gate = (
            torch.nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_id, config.n_layer) else None
        )
        self.monitoring = False
        self.causal = bool(getattr(config, 'causal', True))
        self.latest_metrics = {}

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        ve = kwargs.get("ve")
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve
        if self.config.clip_qkv is not None:
            q = q.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k = k.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v = v.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        if self.config.qk_bias:
            q_bias, k_bias = self.qk_bias.split(1, dim=0)
            q = (q + q_bias).to(q.dtype)
            k = (k + k_bias).to(q.dtype)
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        if self.config.qk_norm:
            q, k = norm(q), norm(k)

        window_size = kwargs.get("window_size", (-1, -1))
        kv_cache = kwargs.get("kv_cache")
        past_key_values = kwargs.get("past_key_values")
        step_idx = kwargs.get("step_idx")

        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_id)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=self.causal,
                window_size=window_size,
            )
            if self.layer_id == kv_cache.n_layers - 1:
                kv_cache.advance(T)
        elif past_key_values is not None and step_idx is not None:
            # ParcaeDynamicCache: cache expects (B, H, S, D) layout
            k_for_cache = k.transpose(1, 2)
            v_for_cache = v.transpose(1, 2)
            k_cached, v_cached = past_key_values.update(k_for_cache, v_for_cache, step_idx)
            k_full = k_cached.transpose(1, 2)
            v_full = v_cached.transpose(1, 2)
            y = flash_attn.flash_attn_func(q, k_full, v_full, causal=self.causal, window_size=window_size)
        else:
            y = flash_attn.flash_attn_func(q, k, v, causal=self.causal, window_size=window_size)

        y = y.contiguous().view(B, T, C)
        return self.c_proj(y)

    @staticmethod
    def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """Repeat K/V heads for GQA: torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
        if n_rep == 1:
            return x
        B, T, H, D = x.shape
        return x.unsqueeze(3).expand(B, T, H, n_rep, D).reshape(B, T, H * n_rep, D)

    @torch.no_grad()
    def monitor_layer(self, q, k, mask):
        """Casting metric computations into low precision because fusion op can be unreliable"""
        S = q.shape[1]
        if mask is None:
            attn_mask_tril = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            attn_mask = torch.zeros_like(attn_mask_tril).to(q)
            attn_mask = attn_mask_tril.masked_fill(~attn_mask_tril, -10000)
        else:
            attn_mask = mask
        q = q.half().transpose(1, 2)  # (B, nh, S, hs)
        k = k.half().transpose(1, 2)
        A = ((q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)) + attn_mask).half()
        max_attn_logit = A.max()
        A = torch.softmax(A, dim=-1)  # overwrite A immediately
        if self.config.center_attention:
            A = A + torch.eye(S, device=A.device, dtype=A.dtype)[None, None, :, :]
        if self.config.debias_attention:
            mask_matrix = torch.ones([S, S], dtype=torch.bool, device=q.device).tril()
            A = A - mask_matrix / mask_matrix.sum(dim=1, keepdim=True)
        attn_entropy = 1 / S * torch.where(A > 0, -A.float() * A.float().log(), 0).sum(dim=-1).sum(dim=-1).mean()
        metrics = {f"attn_entropy_{self.layer_id}": attn_entropy, f"max_attn_logit_{self.layer_id}": max_attn_logit}
        self.latest_metrics = metrics  # will be picked up from monitoring caller
