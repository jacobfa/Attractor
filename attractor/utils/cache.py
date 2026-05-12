from typing import Optional
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


class KVCache(nn.Module):

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        shape = (batch_size, max_seq_len, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype, device=device))
        self.max_seq_len = max_seq_len
        self.seq_len = 0

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = k.shape[1]
        
        if tok_idx is None:
            tok_idx = torch.arange(self.seq_len, self.seq_len + seq_len, device=k.device)
        
        self.k_cache.index_copy_(1, tok_idx, k)
        self.v_cache.index_copy_(1, tok_idx, v)
        
        self.seq_len = max(self.seq_len, tok_idx.max().item() + 1)
        
        return self.k_cache[:, :self.seq_len], self.v_cache[:, :self.seq_len]

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0

    def get_seq_length(self) -> int:
        return self.seq_len


class ParcaeDynamicCache(DynamicCache):

    def __init__(
        self, 
        lookup_strategy: str = "full", 
        core_step_range: Optional[tuple[int, int]] = None,
        n_core: int = 1,
    ) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.value_cache: dict[int, dict[int, torch.Tensor]] = {}
        self.lookup_strategy = lookup_strategy
        self.core_step_range = core_step_range
        self.n_core = n_core
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        step_idx_tensor: torch.Tensor,
        lookup_strategy: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_idx: int = int(step_idx_tensor)
        lookup_strategy = self.lookup_strategy if lookup_strategy is None else lookup_strategy
        
        in_core_block = True
        if self.core_step_range is not None:
            core_start, core_end = self.core_step_range
            in_core_block = core_start <= step_idx < core_end
        else:
            in_core_block = step_idx > 1
        
        if "compress-" in self.lookup_strategy and in_core_block:
            core_start = self.core_step_range[0] if self.core_step_range else 2
            core_end = self.core_step_range[1] if self.core_step_range else step_idx + 1
            core_size = core_end - core_start
            relative_idx = step_idx - core_start
            n_core = self.n_core
            
            recurrence_idx = relative_idx // n_core
            layer_in_rec = relative_idx % n_core
            num_recurrences = core_size // n_core
            
            if "compress-first" in self.lookup_strategy:
                n_keep_rec = int(self.lookup_strategy.split("compress-first")[1])
                if recurrence_idx >= n_keep_rec:
                    new_step_idx = core_start + n_keep_rec * n_core + layer_in_rec
                else:
                    new_step_idx = step_idx
            elif "compress-last" in self.lookup_strategy:
                n_keep_rec = int(self.lookup_strategy.split("compress-last")[1])
                threshold_rec = num_recurrences - n_keep_rec
                if recurrence_idx < threshold_rec:
                    new_step_idx = core_start + layer_in_rec
                else:
                    kept_rec_idx = recurrence_idx - threshold_rec + 1
                    new_step_idx = core_start + kept_rec_idx * n_core + layer_in_rec
            elif "compress-stride" in self.lookup_strategy:
                stride = int(self.lookup_strategy.split("compress-stride")[1])
                aligned_rec = (recurrence_idx // stride)
                new_step_idx = core_start + aligned_rec * n_core + layer_in_rec
            elif "compress-boundaries" in self.lookup_strategy:
                if recurrence_idx == num_recurrences - 1:
                    new_step_idx = core_start + n_core + layer_in_rec
                else:
                    new_step_idx = core_start + layer_in_rec
            elif "compress-s" in self.lookup_strategy:
                compression_stage = int(self.lookup_strategy.split("compress-")[1][1:])
                new_step_idx = (relative_idx % (compression_stage * n_core)) + core_start
            elif "compress-anchor" in self.lookup_strategy:
                if step_idx - core_start < 4 * 8:
                    new_step_idx = step_idx
                else:
                    new_step_idx = 34 + (step_idx - 34) % 4
            else:  # compress-r
                compression_stage = int(self.lookup_strategy.split("compress-")[1][1:])
                new_step_idx = (step_idx - core_start) // compression_stage + core_start
            step_idx = new_step_idx

        if step_idx not in self.key_cache:
            self.key_cache[step_idx] = {}
            self.value_cache[step_idx] = {}

        if step_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        for idx, entry in enumerate(key_states.unbind(dim=-2)):
            if "compress-" not in self.lookup_strategy:
                assert step_idx < 0 or self._seen_tokens - key_states.shape[-2] + idx not in self.key_cache[step_idx]
            self.key_cache[step_idx][self._seen_tokens - key_states.shape[-2] + idx] = entry
        for idx, entry in enumerate(value_states.unbind(dim=-2)):
            self.value_cache[step_idx][self._seen_tokens - value_states.shape[-2] + idx] = entry

        return (
            torch.stack(list(self.key_cache[step_idx].values()), dim=-2),
            torch.stack(list(self.value_cache[step_idx].values()), dim=-2),
        )

    def reset(self) -> None:
        self._seen_tokens = 0
        self.key_cache.clear()
        self.value_cache.clear()

    def get_seq_length(self) -> int:
        return self._seen_tokens


class GPTKVCache:
    def __init__(self, batch_size, n_layers, n_heads, head_dim, max_seq_len, dtype, device):
        self.n_layers = n_layers
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.caches = [
            KVCache(batch_size, max_seq_len, n_heads, head_dim, dtype, device)
            for _ in range(n_layers)
        ]

    def get_layer_cache(self, layer_id):
        cache = self.caches[layer_id]
        return cache.k_cache, cache.v_cache

    def advance(self, seq_len):
        self.cache_seqlens += seq_len

    def get_seq_length(self):
        return int(self.cache_seqlens[0].item())
