import contextlib
import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from attractor.models.attractor.config import AttractorConfig
from attractor.models.attractor.solvers import anderson_solve, get_solver
from attractor.modules.blocks import TransformerPreNormBlock
from attractor.modules.utils import precompute_freqs_cis
from attractor.ops import LinearCrossEntropyLoss


def _maybe_clip_jv(Jv: Tensor, v: Tensor, adjoint_clip) -> Tensor:
    """Rescale J^T v when its per-sample norm exceeds adjoint_clip * ||v||.
    Keeps the Neumann-1 adjoint approximation safe if the head drifts towards
    a less contractive regime during training."""
    if adjoint_clip is None:
        return Jv
    B = Jv.size(0)
    v_norm = v.reshape(B, -1).norm(dim=1).clamp_min(1e-12)
    Jv_norm = Jv.reshape(B, -1).norm(dim=1)
    bound = float(adjoint_clip) * v_norm
    scale = torch.where(Jv_norm > bound,
                        bound / Jv_norm.clamp_min(1e-12),
                        torch.ones_like(Jv_norm))
    return Jv * scale.view(B, *([1] * (Jv.ndim - 1)))


class _IFTContext:
    """Side-channel used by _IFTAttach to hold tensors that carry an autograd
    graph but should not be passed as Function inputs (doing so would cause
    double-backward errors when the outer engine re-traverses them)."""

    def __init__(self, y_out, y_s, bw_kwargs):
        self.y_out = y_out
        self.y_s = y_s
        self.bw_kwargs = bw_kwargs


class _IFTAttach(torch.autograd.Function):
    """Attaches the implicit-function-theorem (IFT) gradient to a fixed point.

    Forward is a no-op (returns y_star_value unchanged). Backward solves the
    adjoint system (I - J_f^T) u = v, then propagates u through f's graph to
    compute gradients for the backbone context c and every attractor-head
    parameter. Routing through the Function input list (rather than
    register_hook) ensures DDP's per-parameter hooks fire on the head
    parameters, which is required for them to receive valid gradients."""

    @staticmethod
    def forward(ctx, y_star_value, iftc: "_IFTContext", c, *fp_params):
        ctx.iftc = iftc
        ctx._n_params = len(fp_params)
        return y_star_value

    @staticmethod
    def backward(ctx, grad_y_star):
        iftc: _IFTContext = ctx.iftc
        y_out, y_s, kw = iftc.y_out, iftc.y_s, iftc.bw_kwargs
        v = grad_y_star.contiguous()

        if kw["bw_type"] == "onestep":
            # Neumann-1 approximation: u = v + J_f^T v
            Jv, = torch.autograd.grad(y_out, y_s, v,
                                      retain_graph=True, create_graph=False)
            Jv = _maybe_clip_jv(Jv, v, kw["adjoint_clip"])
            u = Jv + v
        else:
            def T_op(u):
                Ju, = torch.autograd.grad(y_out, y_s, u,
                                          retain_graph=True, create_graph=False)
                return Ju + v

            if kw["bw_type"] == "anderson":
                with torch.no_grad():
                    u, _ = anderson_solve(
                        T_op, v.detach().clone(),
                        max_iter=kw["bw_max_iter"], tol=kw["bw_tol"],
                        m=kw["anderson_m"], beta=kw["anderson_beta"],
                        min_iter=kw["bw_min_iter"])
            else:
                u = v.detach().clone()
                for it in range(kw["bw_max_iter"]):
                    u_new = T_op(u)
                    diff = (u_new - u).reshape(u.size(0), -1).norm(dim=1)
                    ref = u_new.reshape(u.size(0), -1).norm(dim=1).clamp_min(1e-9)
                    u = u_new
                    if (it + 1) >= kw["bw_min_iter"] and \
                       (diff / ref).max().item() < kw["bw_tol"]:
                        break

        all_inputs = ctx.iftc._inputs_for_grad   # (c, *fp_params)
        diff_targets = tuple(t for t in all_inputs if t.requires_grad)
        if diff_targets and y_out.requires_grad:
            try:
                diff_grads = torch.autograd.grad(
                    y_out, diff_targets, u,
                    retain_graph=False, create_graph=False, allow_unused=True)
            except RuntimeError as e:
                states = ", ".join(
                    f"{i}:rg={t.requires_grad},leaf={t.is_leaf}"
                    for i, t in enumerate(diff_targets))
                raise RuntimeError(
                    f"{e}\n_IFTAttach.backward: y_out.rg="
                    f"{y_out.requires_grad}, n_targets={len(diff_targets)}, "
                    f"states=[{states}]") from e
            grad_lookup = dict(zip(map(id, diff_targets), diff_grads))
        else:
            grad_lookup = {}

        def _grad_for(t):
            g = grad_lookup.get(id(t), None)
            return torch.zeros_like(t) if g is None else g

        c = all_inputs[0]
        c_grad = _grad_for(c) if c.requires_grad else None
        param_grads = tuple(_grad_for(p) for p in all_inputs[1:])
        return (None, None, c_grad) + param_grads


class FixedPointBlock(TransformerPreNormBlock):
    """A single weight-tied transformer block used as the attractor head.

    Applies per-channel LayerScale gating (raw_gamma_*) so the output of each
    sub-layer can be scaled independently. Gates are initialized to
    layer_scale_init so the Jacobian J_f is small at the start of training,
    keeping the solver contractive. The gate can grow up to gamma_max as
    training proceeds."""

    def __init__(self, config, layer_id: int,
                 layer_scale_init: Optional[float] = None,
                 gamma_max: Optional[float] = None) -> None:
        super().__init__(config, layer_id=layer_id)
        n_embd = config.n_embd

        # If the parent init strategy zeros the output projections (e.g.
        # scaled-zero), the attractor map f(y,c) = FPHead(y+c) - y becomes the
        # constant c — the solver trivially converges and gates receive no
        # gradient. We re-init to a non-zero std so the head has real work to
        # do from the first step.
        out_std = math.sqrt(2.0 / (5.0 * n_embd))
        with torch.no_grad():
            for w in (self.attn.c_proj.weight, self.mlp.proj.weight):
                if getattr(w, "is_meta", False):
                    continue
                if w.abs().max().item() == 0.0:
                    nn.init.trunc_normal_(
                        w, mean=0.0, std=out_std,
                        a=-3 * out_std, b=3 * out_std)

        # Value embeddings don't make sense in the iterative head.
        if getattr(self.attn, "ve_gate", None) is not None:
            self.attn.ve_gate = None

        if layer_scale_init is not None:
            gmax = float(gamma_max) if gamma_max is not None else 1.0
            p = max(min(float(layer_scale_init) / gmax, 1.0 - 1e-6), 1e-6)
            raw_init = math.log(p / (1.0 - p))
            self.raw_gamma_attn = nn.Parameter(torch.full((n_embd,), raw_init))
            self.raw_gamma_mlp = nn.Parameter(torch.full((n_embd,), raw_init))
            self.raw_gamma_attn._no_weight_decay = True
            self.raw_gamma_mlp._no_weight_decay = True
            self.gamma_max = gmax
        else:
            self.register_parameter("raw_gamma_attn", None)
            self.register_parameter("raw_gamma_mlp", None)
            self.gamma_max = 1.0

    def forward(self, x: Tensor, freqs_cis: Tensor,
                mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        kwargs.pop("ve", None)
        attn_out = self.attn(self.norm_1(x), freqs_cis, mask, **kwargs)
        if self.raw_gamma_attn is not None:
            attn_out = attn_out * (torch.sigmoid(self.raw_gamma_attn) * self.gamma_max)
        x = x + attn_out
        mlp_out = self.mlp(self.norm_2(x))
        if self.raw_gamma_mlp is not None:
            mlp_out = mlp_out * (torch.sigmoid(self.raw_gamma_mlp) * self.gamma_max)
        x = x + mlp_out
        return x


class Attractor(nn.Module):
    """Attractor language model.

    A causal transformer backbone (prelude) encodes input tokens into context
    c(x). A weight-tied attractor head f_θ is then iterated to a fixed point
    y* in embedding space, with c(x) injected at every step. Logits come from
    weight-tied unembedding of LN(y*).

    The fixed point is located with Anderson acceleration (black-box, no BPTT
    through the solver). Gradients flow back through the implicit function
    theorem via _IFTAttach, keeping training memory O(1) in solver iterations."""

    _default_objective = {"ignore_index": -100, "z_regularization": 0.0}

    def __init__(self, config: AttractorConfig,
                 objective: Optional[dict[str, Any]] = None,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        objective = objective or self._default_objective
        assert config.padded_vocab_size is not None
        self.config = config
        self.objective = objective
        self.emb_scale = config.init.embedding_scale

        prelude = nn.ModuleList(
            config.Block(config, layer_id=i)
            for i in range(config.n_layers_in_prelude)
        )
        core_block = nn.ModuleList(
            FixedPointBlock(
                config,
                layer_id=config.n_layers_in_prelude + i,
                layer_scale_init=config.layer_scale_init,
                gamma_max=config.gamma_max,
            ) for i in range(config.n_layers_in_recurrent_block)
        )
        coda = nn.ModuleList(
            config.Block(
                config,
                layer_id=config.n_layers_in_prelude
                + config.n_layers_in_recurrent_block + i,
            ) for i in range(config.n_layers_in_coda)
        )
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
            prelude=prelude,
            core_block=core_block,
            coda=coda,
            ln_f=config.Norm(config.n_embd, eps=config.norm_eps),
        ))

        # No value embeddings — the head iterates on y, not on raw tokens.
        self.value_embeds = nn.ModuleDict()
        for blk in self.transformer.prelude:
            if getattr(blk.attn, "ve_gate", None) is not None:
                blk.attn.ve_gate = None

        if config.use_fused_head == "full-triton":
            self.lm_head = LinearCrossEntropyLoss(
                config.n_embd,
                config.padded_vocab_size,
                ignore_index=objective["ignore_index"],
                z_regularization=objective["z_regularization"],
                logit_scale=config.init.logit_scale,
                init_method=config.init.fn("head"),
                transposed_weight=not config.tie_embeddings,
            )
        else:
            self.lm_head = config.Linear(
                config.n_embd, config.padded_vocab_size, bias=False,
                init_method=config.init.fn("head"),
            )
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        # Tag the attractor-head parameters so the optimizer can place them in
        # their own LR/WD subgroups (see muon_adamw.get_muon_param_groups).
        fp_wd = float(getattr(config, "fp_wd", 0.0))
        for p in self.transformer.core_block.parameters():
            p._fp_head = True
            p._fp_lr_scale = float(config.fp_lr_scale)
            if p.ndim >= 2 and fp_wd > 0.0:
                p._fp_force_wd = fp_wd
            else:
                p._no_muon_wd = True

        self.max_seq_length = config.block_size
        self.gradient_checkpointing = gradient_checkpointing
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.step = 0
        self.monitoring = False
        self.latest_metrics: dict[str, Any] = {}
        self._last_solver_info: dict[str, Any] = {}

        self.reset_parameters()

    def _precompute_freqs_cis(self) -> Tensor:
        blocks = self.transformer.prelude
        expanded = bool(blocks) and blocks[0].expanded
        dim = self.config.intermediate_size if expanded else self.config.n_embd
        max_length = self.config.randomize_positions_from or self.config.block_size
        return precompute_freqs_cis(
            dim // self.config.num_attention_heads,
            max_length,
            self.config.rope_settings.rope_base,
            self.config.rope_settings.rope_condense_ratio,
        )

    def reset_parameters(self) -> None:
        self.config.init.apply(self.transformer.wte, "embedding")
        self.config.init.apply(self.transformer.ln_f, "normalization")

    @contextlib.contextmanager
    def _deterministic_fp(self):
        """Put the attractor head in eval mode for solver forward passes so
        dropout etc. don't perturb the fixed-point trajectory."""
        was = [blk.training for blk in self.transformer.core_block]
        try:
            for blk in self.transformer.core_block:
                blk.eval()
            yield
        finally:
            for blk, mode in zip(self.transformer.core_block, was):
                blk.train(mode)

    def _encode(self, input_ids: Tensor, freqs_cis: Tensor,
                attention_mask: Optional[Tensor]) -> Tensor:
        x = self.transformer.wte(input_ids)
        if self.emb_scale != 1:
            x = x * self.emb_scale
        for i, block in enumerate(self.transformer.prelude):
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            if self.gradient_checkpointing:
                x = self.config.checkpoint(block, x, freqs_cis, attention_mask, ve=ve)
            else:
                x = block(x, freqs_cis, attention_mask, ve=ve)
        return x

    def _fp_head(self, y: Tensor, freqs_cis: Tensor,
                 attention_mask: Optional[Tensor]) -> Tensor:
        h = y
        for blk in self.transformer.core_block:
            h = blk(h, freqs_cis, attention_mask)
        return h

    def _f(self, y: Tensor, c: Tensor, freqs_cis: Tensor,
           attention_mask: Optional[Tensor]) -> Tensor:
        """The attractor map: f(y, c) = FPHead(y + c) - y.
        The fixed point satisfies y* = f(y*, c), i.e. FPHead(y*+c) = y*+c."""
        return self._fp_head(y + c, freqs_cis, attention_mask) - y

    def _solver_kwargs(self, max_iter_override: Optional[int] = None):
        max_iter = int(max_iter_override) if max_iter_override else self.config.max_iter
        min_iter = min(int(getattr(self.config, "min_iter", 0)), max_iter)
        name = self.config.solver
        if name == "anderson" and max_iter < 3:
            name = "fpi"
        kw = dict(max_iter=max_iter, tol=float(self.config.tol), min_iter=min_iter)
        if name == "anderson":
            kw.update(m=int(self.config.anderson_m),
                      beta=float(self.config.anderson_beta))
        return name, kw

    def _solve_forward(self, c: Tensor, freqs_cis: Tensor,
                       attention_mask: Optional[Tensor],
                       max_iter_override: Optional[int] = None):
        name, kw = self._solver_kwargs(max_iter_override)
        solver = get_solver(name)
        y0 = c.detach()
        with torch.no_grad(), self._deterministic_fp():
            y_star, info = solver(
                lambda y: self._f(y, c.detach(), freqs_cis, attention_mask),
                y0, **kw)
        return y_star, info

    def _attach_implicit_grad(self, y_star_nograd: Tensor, c: Tensor,
                              freqs_cis: Tensor,
                              attention_mask: Optional[Tensor]) -> Tensor:
        y_s = y_star_nograd.detach().requires_grad_(True)
        with self._deterministic_fp():
            y_out = self._f(y_s, c, freqs_cis, attention_mask)
        if not y_out.requires_grad:
            return y_star_nograd.detach()

        fp_params = tuple(p for p in self.transformer.core_block.parameters()
                          if p.requires_grad)
        bw_kwargs = dict(
            bw_type=self.config.backward_type,
            bw_max_iter=int(self.config.backward_max_iter),
            bw_min_iter=min(
                int(getattr(self.config, "backward_min_iter", 0) or 0),
                int(self.config.backward_max_iter)),
            bw_tol=float(self.config.backward_tol),
            anderson_m=int(self.config.anderson_m),
            anderson_beta=float(self.config.anderson_beta),
            adjoint_clip=self.config.adjoint_grad_clip,
        )
        iftc = _IFTContext(y_out, y_s, bw_kwargs)
        iftc._inputs_for_grad = (c,) + fp_params
        return _IFTAttach.apply(y_star_nograd.detach(), iftc, c, *fp_params)

    def _refine(self, c: Tensor, freqs_cis: Tensor,
                attention_mask: Optional[Tensor],
                max_iter_override: Optional[int] = None):
        y_star_ng, info = self._solve_forward(
            c, freqs_cis, attention_mask, max_iter_override=max_iter_override)
        y_star = self._attach_implicit_grad(y_star_ng, c, freqs_cis, attention_mask)
        return y_star, y_star_ng, info

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_logits: bool = False,
        num_steps_pair: Optional[Tensor] = None,
    ) -> dict[str, Optional[Tensor]]:
        if self.config.randomize_positions_from is not None and self.training:
            position_ids = torch.sort(torch.randint(
                0, self.config.randomize_positions_from,
                (input_ids.shape[1],), device=input_ids.device))[0]
        if position_ids is None:
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
        else:
            freqs_cis = self.freqs_cis.index_select(1, position_ids)

        c = self._encode(input_ids, freqs_cis, attention_mask)

        max_iter_override = None
        if num_steps_pair is not None:
            try:
                n0 = int(num_steps_pair[0])
                n1 = int(num_steps_pair[1]) if len(num_steps_pair) > 1 else 0
                max_iter_override = max(n0 + n1, 1)
            except TypeError:
                max_iter_override = int(num_steps_pair)

        y_star, _y_star_ng, info = self._refine(
            c, freqs_cis, attention_mask, max_iter_override=max_iter_override)
        self._last_solver_info = info

        x = self.transformer.ln_f(y_star)
        if self.monitoring:
            self.monitor_module(x, y_star, info, c)

        logits: Optional[Tensor] = None
        if labels is not None:
            if self.config.use_fused_head == "cce":
                from cut_cross_entropy import linear_cross_entropy
                loss = linear_cross_entropy(
                    x * self.config.init.logit_scale, self.lm_head.weight,
                    labels, filter_eps="auto")
            elif self.config.use_fused_head == "full-triton":
                loss = self.lm_head(x * self.config.init.logit_scale, labels)
            else:
                logits = self.lm_head(x).float() * self.config.init.logit_scale
                if self.config.logit_softcap is not None:
                    sc = self.config.logit_softcap
                    logits = sc * torch.tanh(logits / sc)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), labels.view(-1),
                    ignore_index=self.objective.get("ignore_index", -100))
            log_ppl = loss.clone().detach()
        else:
            if self.config.use_fused_head == "full-triton":
                w = self.lm_head.weight
                logits = torch.matmul(
                    x, w.T if self.config.tie_embeddings else w
                ).float() * self.config.init.logit_scale
            else:
                logits = self.lm_head(x).float() * self.config.init.logit_scale
            if self.config.logit_softcap is not None:
                sc = self.config.logit_softcap
                logits = sc * torch.tanh(logits / sc)
            loss = torch.as_tensor(0.0)
            log_ppl = torch.as_tensor(0.0)

        return {"loss": loss, "logits": logits if return_logits else None, "log_ppl": log_ppl}

    @torch.no_grad()
    def monitor_module(self, x: Tensor, y_star: Tensor,
                       info: dict, c: Tensor) -> None:
        x_c = x - x.mean(dim=-1, keepdim=True)
        normed_x = x_c / x_c.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        token_corr = (normed_x @ normed_x.transpose(1, 2)).mean() - 1 / x.shape[1]
        dev = x.device
        B = y_star.size(0)
        delta = (y_star.float() - c.float()).reshape(B, -1).norm(dim=1)
        c_norm = c.float().reshape(B, -1).norm(dim=1).clamp_min(1e-9)
        metrics: dict[str, Tensor] = {
            "last_hidden_token_corr": token_corr,
            "last_hidden_norm": x.norm(dim=-1).mean(),
            "fp_rel_residual": torch.tensor(float(info.get("rel_residual", 0.0)), device=dev),
            "fp_iters": torch.tensor(float(info.get("iters", 0)), device=dev),
            "fp_converged": torch.tensor(float(info.get("converged", False)), device=dev),
            "fp_state_norm": y_star.norm(dim=-1).mean(),
            "fp_context_norm": c.norm(dim=-1).mean(),
            # |y* - c| / |c|: measures how much the attractor head moved the
            # fixed point away from the backbone context. Near zero means the
            # head is a near-identity and the backbone is doing all the work.
            "fp_head_contribution": (delta / c_norm).mean(),
        }
        for i, blk in enumerate(self.transformer.core_block):
            gmax = float(getattr(blk, "gamma_max", 1.0))
            if getattr(blk, "raw_gamma_attn", None) is not None:
                g = torch.sigmoid(blk.raw_gamma_attn.detach().float()) * gmax
                metrics[f"fp_gamma_attn_mean_{i}"] = g.mean()
                metrics[f"fp_gamma_attn_max_{i}"] = g.max()
            if getattr(blk, "raw_gamma_mlp", None) is not None:
                g = torch.sigmoid(blk.raw_gamma_mlp.detach().float()) * gmax
                metrics[f"fp_gamma_mlp_mean_{i}"] = g.mean()
                metrics[f"fp_gamma_mlp_max_{i}"] = g.max()
        self.latest_metrics = metrics


# Keep EQLM as an alias so old checkpoints and eval scripts load without change.
EQLM = Attractor
