import json
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from attractor.models.attractor import Attractor, AttractorConfig
from attractor.utils.cache import GPTKVCache


class ModelingAttractor(Attractor):

    def __init__(self, config: AttractorConfig, objective: Optional[dict] = None,
                 gradient_checkpointing: bool = False) -> None:
        if objective is None:
            objective = {"ignore_index": -100, "z_regularization": 0.0}
        super().__init__(config, objective, gradient_checkpointing)
        self._generation_config = None
        self._compiled = False
        self._reset_solver_stats()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path],
                        device=None, dtype=None, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(
                repo_id=str(pretrained_model_name_or_path),
                allow_patterns=["*.json", "*.bin", "*.safetensors", "*.pt"]))
        from attractor.models.config import RoPESettings
        with open(path / "config.json") as f:
            config_dict = json.load(f)
        if "rope_settings" in config_dict and isinstance(config_dict["rope_settings"], dict):
            config_dict["rope_settings"] = RoPESettings(**config_dict["rope_settings"])
        config_dict.update(kwargs)
        for key in ["_class_name", "init"]:
            config_dict.pop(key, None)
        model = cls(AttractorConfig(**config_dict))
        weights_path = None
        for name in ["pytorch_model.bin", "model.safetensors", "model.bin", "model.pt"]:
            if (path / name).exists():
                weights_path = path / name
                break
        if weights_path is None:
            raise FileNotFoundError(f"No weights found in {path}")
        if weights_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ["module.", "_orig_mod.", "model.", "_forward_module."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v
        model.load_state_dict(cleaned, strict=False)
        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)
        model.eval()
        return model

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config_dict = self.config.__getstate__()
        for key in ["_class_name", "init"]:
            config_dict.pop(key, None)
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def generation_config(self):
        if self._generation_config is None:
            from transformers import GenerationConfig
            self._generation_config = GenerationConfig(
                max_length=self.config.block_size,
                do_sample=True, temperature=1.0)
        return self._generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._generation_config = value

    def apply_eval_solver(self, **overrides) -> None:
        recognised = {
            "solver", "tol", "max_iter", "min_iter",
            "anderson_m", "anderson_beta",
            "backward_tol", "backward_max_iter", "backward_min_iter",
            "adjoint_grad_clip",
        }
        int_keys = {"max_iter", "min_iter", "anderson_m",
                    "backward_max_iter", "backward_min_iter"}
        float_keys = {"tol", "anderson_beta", "backward_tol", "adjoint_grad_clip"}
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in recognised:
                print(f"[ModelingAttractor] WARNING: ignoring unknown eval_solver key {k!r}")
                continue
            if k in int_keys and not isinstance(v, int):
                v = int(v)
            elif k in float_keys and not isinstance(v, float):
                v = float(v)
            setattr(self.config, k, v)
        self._reset_solver_stats()

    def _reset_solver_stats(self) -> None:
        self._solver_stats = {
            "samples": 0, "calls": 0,
            "iters_sum": 0.0, "rel_res_sum": 0.0, "converged_sum": 0.0,
            "max_iters_seen": 0.0, "max_rel_res_seen": 0.0,
            "non_converged_calls": 0,
        }

    def reset_solver_stats(self) -> None:
        self._reset_solver_stats()

    def _record_solver_info(self, info: dict, batch_size: int) -> None:
        if not info:
            return
        s = self._solver_stats
        iters = float(info.get("iters", 0.0))
        rel = float(info.get("rel_residual", 0.0))
        converged = float(bool(info.get("converged", False)))
        s["samples"] += batch_size
        s["calls"] += 1
        s["iters_sum"] += iters * batch_size
        s["rel_res_sum"] += rel * batch_size
        s["converged_sum"] += converged * batch_size
        if iters > s["max_iters_seen"]:
            s["max_iters_seen"] = iters
        if rel > s["max_rel_res_seen"]:
            s["max_rel_res_seen"] = rel
        if converged < 1.0:
            s["non_converged_calls"] += 1

    def solver_summary(self) -> dict:
        s = self._solver_stats
        n = max(s["samples"], 1)
        c = max(s["calls"], 1)
        return {
            "samples": s["samples"], "calls": s["calls"],
            "mean_iters": s["iters_sum"] / n,
            "mean_rel_res": s["rel_res_sum"] / n,
            "frac_converged": s["converged_sum"] / n,
            "max_iters_seen": s["max_iters_seen"],
            "max_rel_res_seen": s["max_rel_res_seen"],
            "non_converged_calls": s["non_converged_calls"],
            "frac_calls_with_nonconv": s["non_converged_calls"] / c,
        }

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_steps": kwargs.get("num_steps"),
        }

    @torch.no_grad()
    def forward_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> dict:
        seq_len = input_ids.shape[1]
        freqs_cis = self.freqs_cis[:, :seq_len]

        c = self._encode(input_ids, freqs_cis, attention_mask)

        max_iter_override = int(num_steps) if num_steps is not None else None
        y_star, info = self._solve_forward(
            c, freqs_cis, attention_mask, max_iter_override=max_iter_override)
        self._last_solver_info = info
        self._record_solver_info(info, batch_size=input_ids.size(0))

        x = self.transformer.ln_f(y_star)

        if self.config.use_fused_head == "full-triton":
            w = self.lm_head.weight
            logits = torch.matmul(
                x, w.T if self.config.tie_embeddings else w
            ).float() * self.config.init.logit_scale
        else:
            logits = self.lm_head(x).float() * self.config.init.logit_scale

        if getattr(self.config, "logit_softcap", None) is not None:
            sc = self.config.logit_softcap
            logits = sc * torch.tanh(logits / sc)

        return {"logits": logits}

    def compile_for_inference(self, mode: str = "default", dynamic: bool = True) -> None:
        # torch.compile every prelude + core block. Idempotent. If compilation
        # later fails at runtime (e.g. cpp toolchain missing), torch's
        # `suppress_errors=True` falls back to eager per-call.
        if self._compiled:
            return
        try:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.suppress_errors = True
        except AttributeError:
            pass
        self.transformer.prelude = nn.ModuleList([
            torch.compile(b, mode=mode, dynamic=dynamic)
            for b in self.transformer.prelude
        ])
        self.transformer.core_block = nn.ModuleList([
            torch.compile(b, mode=mode, dynamic=dynamic)
            for b in self.transformer.core_block
        ])
        self._compiled = True

    def create_cache(self, batch_size: int, max_seq_len: Optional[int] = None,
                     dtype: Optional[torch.dtype] = None,
                     device: Optional[torch.device] = None) -> GPTKVCache:
        # Cache only spans the prelude. The FP head is re-run over full c each step.
        if max_seq_len is None:
            max_seq_len = int(self.config.block_size)
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if device is None:
            device = self.device
        head_dim = self.config.n_embd // self.config.num_attention_heads
        return GPTKVCache(
            batch_size=batch_size,
            n_layers=len(self.transformer.prelude),
            n_heads=self.config.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )

    def _encode_cached(self, input_ids: torch.Tensor,
                       kv_cache: GPTKVCache) -> torch.Tensor:
        T = input_ids.shape[1]
        cache_len = kv_cache.get_seq_length()
        freqs_cis = self.freqs_cis[:, cache_len:cache_len + T]
        x = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        for block in self.transformer.prelude:
            x = block(x, freqs_cis, None, kv_cache=kv_cache)
        return x

    def _logits_from_state(self, y_star: torch.Tensor) -> torch.Tensor:
        x = self.transformer.ln_f(y_star)
        if self.config.use_fused_head == "full-triton":
            w = self.lm_head.weight
            logits = torch.matmul(
                x, w.T if self.config.tie_embeddings else w
            ).float() * self.config.init.logit_scale
        else:
            logits = self.lm_head(x).float() * self.config.init.logit_scale
        if getattr(self.config, "logit_softcap", None) is not None:
            sc = self.config.logit_softcap
            logits = sc * torch.tanh(logits / sc)
        return logits

    def _sample_from_logits(self, logits, temperature, top_k, top_p, do_sample):
        logits = logits / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative > top_p
            mask[:, 1:] = mask[:, :-1].clone()
            mask[:, 0] = False
            logits[mask.scatter(1, sorted_indices, mask)] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return (torch.multinomial(probs, num_samples=1)
                if do_sample else logits.argmax(dim=-1, keepdim=True))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_steps: Optional[int] = None,
        use_cache: bool = True,
        streamer=None,
        **kwargs,
    ) -> torch.Tensor:
        # Cached path: stash prelude KV per layer and stash full c across steps;
        # FP head still runs over the full c each step (it's weight-tied and
        # iter-dependent on the recurrent state, so we can't cache its output).
        block_size = int(self.config.block_size)
        max_iter_override = int(num_steps) if num_steps is not None else None
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        if not use_cache:
            return self._generate_no_cache(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                do_sample=do_sample, num_steps=num_steps, streamer=streamer,
            )

        cache = self.create_cache(batch_size=input_ids.size(0),
                                  max_seq_len=block_size,
                                  dtype=dtype, device=device)
        c = self._encode_cached(input_ids, cache)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            T_now = c.size(1)
            freqs_cis = self.freqs_cis[:, :T_now]
            y_star, info = self._solve_forward(
                c, freqs_cis, None, max_iter_override=max_iter_override)
            self._last_solver_info = info
            self._record_solver_info(info, batch_size=c.size(0))
            logits = self._logits_from_state(y_star[:, -1:]).squeeze(1)
            next_token = self._sample_from_logits(
                logits, temperature, top_k, top_p, do_sample)
            generated = torch.cat([generated, next_token], dim=-1)
            if streamer is not None:
                streamer.put(next_token.cpu())

            # If sliding past block_size, drop cache and rebuild from the tail.
            if generated.size(1) >= block_size:
                cache = self.create_cache(batch_size=input_ids.size(0),
                                          max_seq_len=block_size,
                                          dtype=dtype, device=device)
                c = self._encode_cached(generated[:, -block_size:], cache)
            else:
                c_new = self._encode_cached(next_token, cache)
                c = torch.cat([c, c_new], dim=1)

        if streamer is not None:
            streamer.end()
        return generated

    @torch.no_grad()
    def _generate_no_cache(self, input_ids, *, max_new_tokens, temperature,
                           top_k, top_p, do_sample, num_steps, streamer):
        block_size = int(self.config.block_size)
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            ctx = generated[:, -block_size:]
            out = self.forward_for_generation(ctx, num_steps=num_steps)
            logits = out["logits"][:, -1, :]
            next_token = self._sample_from_logits(
                logits, temperature, top_k, top_p, do_sample)
            generated = torch.cat([generated, next_token], dim=-1)
            if streamer is not None:
                streamer.put(next_token.cpu())
        if streamer is not None:
            streamer.end()
        return generated
