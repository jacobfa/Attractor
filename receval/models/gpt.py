import torch
from typing import Optional

from attractor.models.gpt import GPT, GPTConfig
from attractor.utils.cache import GPTKVCache


class ModelingGPT(GPT):

    def __init__(self, config: GPTConfig, objective=None, gradient_checkpointing=False) -> None:
        if objective is None:
            objective = {"ignore_index": -100, "z_regularization": 0.0}
        super().__init__(config, objective, gradient_checkpointing)
        self._generation_config = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def generation_config(self):
        if self._generation_config is None:
            from transformers import GenerationConfig
            self._generation_config = GenerationConfig(
                max_length=self.config.block_size,
                do_sample=True,
                temperature=1.0,
            )
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    def create_cache(self, batch_size: int, max_seq_len: Optional[int] = None, dtype=None, device=None):
        if max_seq_len is None:
            max_seq_len = self.config.block_size
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = self.device
        head_dim = self.config.n_embd // self.config.num_attention_heads
        return GPTKVCache(
            batch_size=batch_size,
            n_layers=self.config.n_layer,
            n_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[GPTKVCache] = None,
        **kwargs,
    ) -> dict:
        seq_len = input_ids.shape[1]
        cache_len = kv_cache.get_seq_length() if kv_cache is not None else 0
        freqs_cis = self.freqs_cis[:, cache_len:cache_len + seq_len]

        x = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            x = x * self.emb_scale

        for i, block in enumerate(self.transformer.h):
            ve_embed = self.value_embeds[str(i)] if str(i) in self.value_embeds else None
            ve = ve_embed(input_ids) if ve_embed is not None else None
            x = block(x, freqs_cis, attention_mask, ve=ve, kv_cache=kv_cache)

        x = self.transformer.ln_f(x)

        if self.config.tie_embeddings:
            logits = torch.matmul(x, self.transformer.wte.weight.T).float()
        else:
            logits = torch.matmul(x, self.lm_head.weight).float()

        if getattr(self.config, 'logit_softcap', None) is not None:
            softcap = self.config.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)

        return {"logits": logits, "kv_cache": kv_cache}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_cache: bool = True,
        streamer=None,
        **kwargs,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        batch_size = input_ids.shape[0]
        kv_cache = self.create_cache(batch_size) if use_cache else None
        generated = input_ids.clone()
        current_input = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward_for_generation(current_input, kv_cache=kv_cache)
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                logits[mask.scatter(1, sorted_indices, mask)] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if do_sample else logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            current_input = next_token if use_cache else generated

            if streamer is not None:
                streamer.put(next_token.cpu())

        if streamer is not None:
            streamer.end()

        return generated
