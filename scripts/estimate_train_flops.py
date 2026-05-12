"""Analytic training-FLOPs estimator for parcae and EQLM at every model size.

This is a self-contained closed-form calculator (no model construction needed).
It accounts for:

  * Forward (2N rule) + backward (4N rule) matmul FLOPs per parameter
  * Attention forward (4·S²·D per layer per token) + backward (2× forward)
  * Recurrent-core unroll cost depending on backward strategy:
        - parcae_bptt   : full BPTT through all T iterations  (default Parcae)
        - parcae_1step  : 1-step gradient (HRM-style; T-1 fwd no-grad + last fwd+bwd)
        - eqlm_ift      : EQLM IFT 1-step JVP backward (T fwd no-grad + 1 fwd+bwd
                          for the implicit-gradient hook, plus 1 JVP)
        - eqlm_anderson : EQLM with Anderson backward (B_iter JVPs, each ~ 1 fwd cost)
  * Activation recomputation (gradient checkpointing):
        - "none"      : no recompute
        - "all"       : recompute every transformer block (extra fwd per block)
        - "attn"      : recompute attention only (cheap; adds ~33% to attn fwd cost)
  * Optimizer overhead:
        - AdamW: ~8 FLOPs per parameter per optimizer step (2 momentums + step).
                 Counted as `8 · N_total · num_optim_steps` total.

The Parcae and EQLM configs are pinned to the values registered in
``attractor/configs/{parcae,eqlm}/*-{small,medium,large}.py`` (n_embd,
intermediate_size, layer counts, vocab_size, padded vocab) so the formula
doesn't drift if a config is edited.

Token budgets default to the ``max_tokens`` field in
``launch_configs/{parcae,eqlm}-{small,medium,large}-*.yaml``.

Usage:
    python scripts/estimate_train_flops.py
    python scripts/estimate_train_flops.py --t_eqlm 6 --t_parcae 8 \\
                                            --bptt full --recompute none
    python scripts/estimate_train_flops.py --json out.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Literal, Optional


# --------------------------------------------------------------------------- #
# Pinned arch + token budgets (kept in sync with attractor/configs/* and
# launch_configs/*).
# --------------------------------------------------------------------------- #

@dataclass
class ModelArch:
    name: str
    n_embd: int
    intermediate_size: int
    num_attention_heads: int
    n_prelude: int
    n_core: int
    n_coda: int
    vocab_size: int = 32768
    block_size: int = 2048


PARCAE: Dict[str, ModelArch] = {
    "small":  ModelArch("parcae-small-140m",  n_embd=1024, intermediate_size=4096,
                        num_attention_heads=8,  n_prelude=2, n_core=2, n_coda=2),
    "medium": ModelArch("parcae-medium-370m", n_embd=1280, intermediate_size=5120,
                        num_attention_heads=10, n_prelude=4, n_core=4, n_coda=4),
    "large":  ModelArch("parcae-large-770m",  n_embd=1536, intermediate_size=6144,
                        num_attention_heads=12, n_prelude=8, n_core=8, n_coda=8),
}

EQLM: Dict[str, ModelArch] = {
    "small":  ModelArch("eqlm-small-140m",  n_embd=1024, intermediate_size=4096,
                        num_attention_heads=8,  n_prelude=7,  n_core=1, n_coda=0),
    "medium": ModelArch("eqlm-medium-370m", n_embd=1280, intermediate_size=5120,
                        num_attention_heads=10, n_prelude=15, n_core=2, n_coda=0),
    "large":  ModelArch("eqlm-large-770m",  n_embd=1536, intermediate_size=6144,
                        num_attention_heads=12, n_prelude=35, n_core=2, n_coda=0),
}

TRAIN_TOKENS: Dict[str, float] = {
    "small":  11.2e9,
    "medium": 29.6e9,
    "large":  61.6e9,
}


# --------------------------------------------------------------------------- #
# Closed-form FLOP accounting
# --------------------------------------------------------------------------- #

def block_matmul_params(n_embd: int, intermediate_size: int) -> int:
    """Pre-norm transformer block matmul params: attn (q+k+v+o) + mlp (fc + proj).
    Norms (RMSNorm scale) are negligible; we ignore them."""
    attn = 4 * n_embd * n_embd
    mlp = 2 * n_embd * intermediate_size
    return attn + mlp


def attn_fwd_per_token(seq_len: int, n_embd: int) -> int:
    """FlashAttention-style forward FLOPs *per token per layer*: ~4·S·D.
    (Per-batch is 4·S²·D; dividing by S gives per-token.)"""
    return 4 * seq_len * n_embd


@dataclass
class TrainCostConfig:
    # Recurrence iterations
    t_eqlm: int = 6
    t_parcae: int = 8

    # Parcae backward: 'full' (BPTT) or '1step' (HRM-style 1-step gradient)
    parcae_bptt: Literal["full", "1step"] = "full"

    # EQLM backward: 'ift' (1-step JVP) or 'anderson' (B_iter JVPs)
    eqlm_bw: Literal["ift", "anderson"] = "ift"
    # When eqlm_bw='anderson', this many JVPs in the adjoint solve:
    eqlm_bw_iters: int = 4

    # Activation recompute policy: 'none', 'all', 'attn'
    recompute: Literal["none", "all", "attn"] = "none"

    # Sequence length used in attention FLOP accounting
    seq_len: int = 2048

    # Optimizer overhead per parameter per optimizer step (FLOPs)
    # AdamW: ~8-10 ops per param per step (2 momentum updates, bias correction,
    # final update). 8 is a common conservative number.
    optim_flops_per_param: int = 8

    # Effective batch (tokens per optimizer step) for optimizer overhead
    # accounting. parcae's typical setting:
    tokens_per_optim_step: int = 1_048_576  # 1M tokens / step (parcae default)


@dataclass
class FlopsBreakdown:
    """All FLOP terms separately, in 'per-training-token' units."""
    non_core_matmul_fwd: float
    non_core_matmul_bwd: float
    non_core_attn_fwd: float
    non_core_attn_bwd: float
    core_matmul_fwd: float
    core_matmul_bwd: float
    core_attn_fwd: float
    core_attn_bwd: float
    recompute_extra: float

    def total_per_token(self) -> float:
        return (
            self.non_core_matmul_fwd + self.non_core_matmul_bwd
            + self.non_core_attn_fwd + self.non_core_attn_bwd
            + self.core_matmul_fwd + self.core_matmul_bwd
            + self.core_attn_fwd + self.core_attn_bwd
            + self.recompute_extra
        )


def _flops_breakdown(arch: ModelArch, *, T: int, kind: Literal["parcae", "attractor"],
                     cfg: TrainCostConfig) -> FlopsBreakdown:
    """Per-training-token FLOPs breakdown for one model+strategy."""
    p_block = block_matmul_params(arch.n_embd, arch.intermediate_size)
    p_emb = arch.vocab_size * arch.n_embd  # tied wte+lm_head, count once
    n_outer = arch.n_prelude + arch.n_coda  # always full fwd+bwd

    # Non-core (prelude+coda+embed+head) -----------------------------------
    nc_matmul_fwd = 2 * (p_emb + p_block * n_outer)
    nc_matmul_bwd = 4 * (p_emb + p_block * n_outer)
    attn_per_layer = attn_fwd_per_token(cfg.seq_len, arch.n_embd)
    nc_attn_fwd = attn_per_layer * n_outer
    nc_attn_bwd = 2 * attn_per_layer * n_outer  # bwd ≈ 2× fwd

    # Core (recurrent T iterations) ----------------------------------------
    if kind == "parcae":
        if cfg.parcae_bptt == "full":
            # BPTT: every iteration is a fwd (2N) + bwd (4N).
            c_mm_fwd = 2 * p_block * arch.n_core * T
            c_mm_bwd = 4 * p_block * arch.n_core * T
            c_at_fwd = attn_per_layer * arch.n_core * T
            c_at_bwd = 2 * attn_per_layer * arch.n_core * T
        else:  # parcae_bptt == "1step"
            # T-1 fwd no-grad + 1 fwd+bwd through last iter (HRM 1-step gradient)
            c_mm_fwd = 2 * p_block * arch.n_core * T
            c_mm_bwd = 4 * p_block * arch.n_core * 1
            c_at_fwd = attn_per_layer * arch.n_core * T
            c_at_bwd = 2 * attn_per_layer * arch.n_core * 1
    else:  # kind == "attractor"
        if cfg.eqlm_bw == "ift":
            # T no-grad fwds + 1 fwd-with-grad (for IFT setup) + 1 JVP (≈ 1 fwd cost)
            c_mm_fwd = 2 * p_block * arch.n_core * (T + 1)
            c_mm_bwd = 2 * p_block * arch.n_core * 1  # JVP ≈ 1 fwd
            c_at_fwd = attn_per_layer * arch.n_core * (T + 1)
            c_at_bwd = attn_per_layer * arch.n_core * 1
        else:  # eqlm_bw == "anderson"
            # T no-grad fwds (forward solve) + B_iter JVPs (adjoint solve);
            # each JVP ≈ 1 fwd cost.
            B = cfg.eqlm_bw_iters
            c_mm_fwd = 2 * p_block * arch.n_core * T
            c_mm_bwd = 2 * p_block * arch.n_core * B
            c_at_fwd = attn_per_layer * arch.n_core * T
            c_at_bwd = attn_per_layer * arch.n_core * B

    # Activation recomputation (extra forward pass for each checkpointed layer)
    if cfg.recompute == "none":
        recompute_extra = 0.0
    elif cfg.recompute == "all":
        # +1 full forward of every checkpointed (= every) block per backward.
        # Cost = 2*p_block*(n_outer + n_core*T_grad) + attn_fwd*(n_outer + n_core*T_grad)
        # where T_grad is the *grad-bearing* iteration count.
        if kind == "parcae":
            T_grad = T if cfg.parcae_bptt == "full" else 1
        else:
            T_grad = 1 if cfg.eqlm_bw == "ift" else cfg.eqlm_bw_iters
        recompute_extra = (
            2 * p_block * (n_outer + arch.n_core * T_grad)
            + attn_per_layer * (n_outer + arch.n_core * T_grad)
        )
    elif cfg.recompute == "attn":
        # Recompute attention only on backward (~ extra attn_fwd on every layer that has grad)
        if kind == "parcae":
            T_grad = T if cfg.parcae_bptt == "full" else 1
        else:
            T_grad = 1 if cfg.eqlm_bw == "ift" else cfg.eqlm_bw_iters
        recompute_extra = attn_per_layer * (n_outer + arch.n_core * T_grad)
    else:
        raise ValueError(f"unknown recompute={cfg.recompute!r}")

    return FlopsBreakdown(
        non_core_matmul_fwd=nc_matmul_fwd,
        non_core_matmul_bwd=nc_matmul_bwd,
        non_core_attn_fwd=nc_attn_fwd,
        non_core_attn_bwd=nc_attn_bwd,
        core_matmul_fwd=c_mm_fwd,
        core_matmul_bwd=c_mm_bwd,
        core_attn_fwd=c_at_fwd,
        core_attn_bwd=c_at_bwd,
        recompute_extra=recompute_extra,
    )


def _total_param_count(arch: ModelArch) -> int:
    p_block = block_matmul_params(arch.n_embd, arch.intermediate_size)
    p_emb = arch.vocab_size * arch.n_embd  # tied embed
    return p_emb + p_block * (arch.n_prelude + arch.n_core + arch.n_coda)


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #

@dataclass
class SizeRow:
    size: str
    tokens: float
    parcae_params: int
    eqlm_params: int
    parcae_flops_per_tok: float
    eqlm_flops_per_tok: float
    parcae_total_flops: float
    eqlm_total_flops: float
    parcae_optim_flops: float
    eqlm_optim_flops: float
    parcae_grand_total: float
    eqlm_grand_total: float
    parcae_breakdown: dict = field(default_factory=dict)
    eqlm_breakdown: dict = field(default_factory=dict)


def _humanise(x: float) -> str:
    """Convert a FLOPs value into a human-readable EFLOP/ZFLOP string."""
    if x >= 1e21:
        return f"{x/1e21:.3f} ZFLOP"
    if x >= 1e18:
        return f"{x/1e18:.2f} EFLOP"
    if x >= 1e15:
        return f"{x/1e15:.1f} PFLOP"
    return f"{x:.2e} FLOP"


def estimate_all(cfg: TrainCostConfig,
                 train_tokens: Optional[Dict[str, float]] = None) -> Dict[str, SizeRow]:
    train_tokens = train_tokens or TRAIN_TOKENS
    out: Dict[str, SizeRow] = {}

    for size in ("small", "medium", "large"):
        p_arch = PARCAE[size]
        e_arch = EQLM[size]
        p_break = _flops_breakdown(p_arch, T=cfg.t_parcae, kind="parcae", cfg=cfg)
        e_break = _flops_breakdown(e_arch, T=cfg.t_eqlm, kind="attractor", cfg=cfg)

        toks = train_tokens[size]
        p_per = p_break.total_per_token()
        e_per = e_break.total_per_token()
        p_total = p_per * toks
        e_total = e_per * toks

        # Optimizer overhead
        p_params = _total_param_count(p_arch)
        e_params = _total_param_count(e_arch)
        n_optim_steps = toks / cfg.tokens_per_optim_step
        p_optim = cfg.optim_flops_per_param * p_params * n_optim_steps
        e_optim = cfg.optim_flops_per_param * e_params * n_optim_steps

        out[size] = SizeRow(
            size=size,
            tokens=toks,
            parcae_params=p_params,
            eqlm_params=e_params,
            parcae_flops_per_tok=p_per,
            eqlm_flops_per_tok=e_per,
            parcae_total_flops=p_total,
            eqlm_total_flops=e_total,
            parcae_optim_flops=p_optim,
            eqlm_optim_flops=e_optim,
            parcae_grand_total=p_total + p_optim,
            eqlm_grand_total=e_total + e_optim,
            parcae_breakdown=asdict(p_break),
            eqlm_breakdown=asdict(e_break),
        )
    return out


def _print_table(results: Dict[str, SizeRow], cfg: TrainCostConfig) -> None:
    print(f"\n{'=' * 120}")
    print(f"Training FLOPs estimate "
          f"(EQLM T={cfg.t_eqlm} {cfg.eqlm_bw}-backward, "
          f"Parcae T={cfg.t_parcae} {cfg.parcae_bptt}-BPTT, "
          f"recompute={cfg.recompute})")
    print(f"{'=' * 120}\n")

    # Per-token comparison
    print(f"{'size':<8}{'params (M)':>14}  "
          f"{'parcae fwd+bwd FLOP/tok':>26}  {'eqlm fwd+bwd FLOP/tok':>24}  {'eqlm/parcae':>12}")
    print("-" * 92)
    for size, r in results.items():
        ratio = r.eqlm_flops_per_tok / r.parcae_flops_per_tok
        print(f"{size:<8}P{r.parcae_params/1e6:5.0f}M E{r.eqlm_params/1e6:5.0f}M  "
              f"{r.parcae_flops_per_tok:>26.3e}  "
              f"{r.eqlm_flops_per_tok:>24.3e}  "
              f"{ratio:>11.2f}x")

    # Total training FLOPs
    print()
    print(f"{'size':<8}{'tokens':>10}  "
          f"{'parcae compute':>20}  {'+ optim':>20}  {'parcae total':>22}  "
          f"{'eqlm total':>20}  {'eqlm/parcae':>12}")
    print("-" * 130)
    for size, r in results.items():
        ratio = r.eqlm_grand_total / r.parcae_grand_total
        print(f"{size:<8}{r.tokens:>9.1e}  "
              f"{_humanise(r.parcae_total_flops):>20}  "
              f"{_humanise(r.parcae_optim_flops):>20}  "
              f"{_humanise(r.parcae_grand_total):>22}  "
              f"{_humanise(r.eqlm_grand_total):>20}  "
              f"{ratio:>11.2f}x")

    # Sanity: for a regular dense model we expect ~6N·D FLOPs total.
    # Print 6N·D as a reference column for parcae's small (T=8) which is most
    # comparable to a dense transformer scaled up by T.
    print(f"\nReference 6N·D for parcae (no recurrence): "
          f"{', '.join(size + ': ' + _humanise(6 * r.parcae_params * r.tokens) for size, r in results.items())}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t_eqlm", type=int, default=6,
                   help="EQLM forward solver iteration count")
    p.add_argument("--t_parcae", type=int, default=8,
                   help="Parcae recurrence count (mean_recurrence)")
    p.add_argument("--bptt", choices=["full", "1step"], default="full",
                   help="Parcae backward strategy")
    p.add_argument("--eqlm_bw", choices=["ift", "anderson"], default="ift",
                   help="EQLM backward strategy")
    p.add_argument("--eqlm_bw_iters", type=int, default=4,
                   help="EQLM Anderson backward iteration count (only used if --eqlm_bw=anderson)")
    p.add_argument("--recompute", choices=["none", "all", "attn"], default="none",
                   help="Activation recomputation policy")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--optim_flops_per_param", type=int, default=8,
                   help="FLOPs per parameter per optimizer step (AdamW ~ 8)")
    p.add_argument("--tokens_per_optim_step", type=int, default=1_048_576)
    p.add_argument("--json", default=None,
                   help="Optional path: also dump full results dict as JSON")
    args = p.parse_args()

    cfg = TrainCostConfig(
        t_eqlm=args.t_eqlm,
        t_parcae=args.t_parcae,
        parcae_bptt=args.bptt,
        eqlm_bw=args.eqlm_bw,
        eqlm_bw_iters=args.eqlm_bw_iters,
        recompute=args.recompute,
        seq_len=args.seq_len,
        optim_flops_per_param=args.optim_flops_per_param,
        tokens_per_optim_step=args.tokens_per_optim_step,
    )

    results = estimate_all(cfg)
    _print_table(results, cfg)

    if args.json:
        payload = {
            "config": asdict(cfg),
            "tokens": TRAIN_TOKENS,
            "parcae": {k: asdict(PARCAE[k]) for k in PARCAE},
            "attractor": {k: asdict(EQLM[k]) for k in EQLM},
            "results": {k: asdict(v) for k, v in results.items()},
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
