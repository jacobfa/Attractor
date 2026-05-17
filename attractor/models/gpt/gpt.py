from typing import Any, Optional

import torch
from torch import Tensor

from attractor.models.gpt.config import GPTConfig
from attractor.ops import LinearCrossEntropyLoss
from attractor.modules.utils import precompute_freqs_cis
from attractor.modules.mixer import has_ve


class GPT(torch.nn.Module):
    _default_objective = {"ignore_index": -100, "z_regularization": 0.0}

    def __init__(
        self,
        config: GPTConfig,
        objective: Optional[dict[str, Any]] = None,
        gradient_checkpointing=False,
    ) -> None:
        super().__init__()
        objective = objective or self._default_objective
        assert config.padded_vocab_size is not None
        self.config = config
        self.emb_scale = config.init.embedding_scale
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.padded_vocab_size, config.n_embd),
                abacus=(
                    torch.nn.Embedding(config.block_size, config.n_embd)
                    if self.config.use_abacus
                    else torch.nn.Identity()
                ),
                h=torch.nn.ModuleList(config.Block(config, layer_id=i) for i in range(config.n_layer)),
                ln_f=config.Norm(config.n_embd, eps=config.norm_eps),
            )
        )
        head_dim = config.n_embd // config.num_attention_heads
        kv_dim = config.num_key_value_heads * head_dim
        self.value_embeds = torch.nn.ModuleDict({
            str(i): torch.nn.Embedding(config.padded_vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.objective = objective
        if self.config.use_fused_head:
            self.lm_head = LinearCrossEntropyLoss(
                config.n_embd,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                z_regularization=objective["z_regularization"],
                logit_scale=config.init.logit_scale,
                init_method=config.init.fn("head"),
                transposed_weight=not self.config.tie_embeddings,
            )
            if self.config.tie_embeddings:
                self.lm_head.weight = self.transformer.wte.weight
        else:
            self.lm_head = config.Linear(
                config.padded_vocab_size, config.n_embd, bias=False, init_method=config.init.fn("head")
            )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[Tensor] = None
        self.gradient_checkpointing = gradient_checkpointing

        self.step = 0
        self.monitoring = False
        self.latest_metrics = {}

        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.reset_parameters()

    def _precompute_freqs_cis(self):
        dim = self.config.intermediate_size if self.transformer.h[0].expanded else self.config.n_embd
        if self.config.randomize_positions_from is not None:
            max_length = self.config.randomize_positions_from
        else:
            max_length = self.config.block_size
        freqs_cis = precompute_freqs_cis(
            dim // self.config.num_attention_heads,
            max_length,
            self.config.rope_settings.rope_base,  # 50k in the newer models
            self.config.rope_settings.rope_condense_ratio,
        )
        return freqs_cis

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> dict[str, Optional[torch.Tensor]]:
        if self.config.randomize_positions_from is not None and self.training:
            position_ids = torch.sort(
                torch.randint(0, self.config.randomize_positions_from, (input_ids.shape[1],), device=input_ids.device)
            )[0]

        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)


        x = self.transformer.wte(input_ids)  # token embeddings of shape (b, t, n_embd)

        if self.emb_scale != 1:
            x = x * self.emb_scale

        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            if not self.gradient_checkpointing:
                x = block(x, freqs_cis, attention_mask, ve=ve)
            else:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask, ve=ve)
        x = self.transformer.ln_f(x)
        if self.monitoring:
            self.monitor_module(x)

        if labels is not None:
            if self.config.use_fused_head:
                loss = self.lm_head(x, labels)
            else:
                if self.config.tie_embeddings:
                    logits = torch.matmul(x, self.transformer.wte.weight.T)
                else:
                    logits = torch.matmul(x, self.lm_head.weight)
                logits = logits.float()
                if self.config.logit_softcap is not None:
                    softcap = self.config.logit_softcap
                    logits = softcap * torch.tanh(logits / softcap)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), labels.view(-1),
                    ignore_index=self.objective.get("ignore_index", -100)
                )
        else:
            if self.config.tie_embeddings:
                outputs = torch.matmul(x, self.transformer.wte.weight.T).float()
            else:
                outputs = torch.matmul(x, self.lm_head.weight).float()
            if self.config.logit_softcap is not None:
                softcap = self.config.logit_softcap
                outputs = softcap * torch.tanh(outputs / softcap)
            loss = torch.as_tensor(0.0)
        return {
            "loss": loss,
            "logits": outputs if return_logits else None,
            "log_ppl": loss.detach(),
        }

    @torch.no_grad()
    def monitor_module(self, x: torch.Tensor):
        x_c = x - x.mean(dim=-1, keepdim=True)
        normed_x = x_c / x_c.norm(dim=-1, keepdim=True)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x.shape[1]
        metrics = {"last_hidden_token_corr": token_corr, "last_hidden_norm": x.norm(dim=-1).mean()}
        self.latest_metrics = metrics

