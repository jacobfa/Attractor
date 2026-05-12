"""Side-by-side FLOPs + latency comparison: Parcae-140M HF vs our EQLM run.

Loads:
  * attractor.from_pretrained("SandyResearch/parcae-140m")  -- pretrained Parcae-140M
  * /scratch1/feinashl/eqlm/eqlm-small-140m  -- our EQLM run dir

Reports for each model:
  - parameter counts (total / prelude / core_block / coda / embedding)
  - analytical training FLOPs/token (recpre.cost.estimate_flops_recurrent)
  - empirical inference FLOPs/token (torch.utils.flop_counter.FlopCounterMode)
  - measured forward-pass latency (cuda events; warmup + N timed steps)
  - tokens/sec
"""
import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from typing import Optional

import torch

# Make torch.compile resilient: capture .item() into the graph (avoids the
# graph-break inside flash_attn_with_kvcache), and fall back to eager on any
# compile error (e.g. a missing cc1plus in the conda toolchain) instead of
# crashing the whole script.
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, str(Path(__file__).parent.parent))

import attractor
from receval.settings import CLISettings
from receval.models.eqlm import ModelingEQLM
from scripts.cost import estimate_flops_recurrent, estimate_flops_gpt


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod.", "model.", "_forward_module."):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def load_parcae_hf(repo_id: str, device: str, dtype: torch.dtype):
    # Use ModelingParcae so we get generate / forward_for_generation /
    # create_cache. The bare Parcae class returned by attractor.from_pretrained
    # is the training shell and has no generate method.
    from receval.models.parcae import ModelingParcae
    print(f"[parcae] Loading {repo_id} ...")
    t0 = time.time()
    model = ModelingParcae.from_pretrained(repo_id, device=device, dtype=dtype)
    model.eval()
    print(f"[parcae] Loaded in {time.time() - t0:.1f}s")
    return model


def load_eqlm_run(out_dir: str, device: str, dtype: torch.dtype):
    print(f"[eqlm] Loading run dir: {out_dir} ...")
    t0 = time.time()
    s = CLISettings(out_dir=out_dir, eval_tasks="bpb")
    if s.checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found under {out_dir}")
    print(f"[eqlm] Checkpoint: {s.checkpoint_path}")
    print(f"[eqlm] model_name: {s.model_name}  model_impl: {s.model_impl}")
    model = ModelingEQLM(s.model_config)
    raw = torch.load(s.checkpoint_path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw)
    sd = _strip_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[eqlm] WARN missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[eqlm] WARN unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[eqlm] Loaded in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def param_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    transformer = getattr(model, "transformer", None)
    out = {"total": total}
    if transformer is not None:
        if hasattr(transformer, "wte"):
            out["wte"] = transformer.wte.weight.numel()
        if hasattr(transformer, "prelude"):
            out["prelude"] = sum(p.numel() for p in transformer.prelude.parameters())
        if hasattr(transformer, "core_block"):
            out["core_block"] = sum(p.numel() for p in transformer.core_block.parameters())
        if hasattr(transformer, "coda"):
            out["coda"] = sum(p.numel() for p in transformer.coda.parameters())
        if hasattr(transformer, "h"):
            out["h"] = sum(p.numel() for p in transformer.h.parameters())
    return out


def analytical_flops(model, num_steps: Optional[int] = None) -> dict:
    cfg = model.config
    if hasattr(cfg, "mean_recurrence"):
        bd = estimate_flops_recurrent(model, cfg)
    else:
        bd = estimate_flops_gpt(model, cfg)
    if num_steps is not None and bd.is_recurrent:
        bd.mean_recurrence = int(num_steps)
        bd.effective_forward_steps = float(max(bd.mean_recurrence - bd.mean_backprop_depth, 0))
    return {
        "is_recurrent": bd.is_recurrent,
        "mean_recurrence": bd.mean_recurrence,
        "mean_backprop_depth": bd.mean_backprop_depth,
        "core_block_params": bd.core_block_params,
        "non_core_params": bd.non_core_params,
        "flops_per_token_train": bd.flops_per_token(use_curriculum_adjusted=False),
    }


@torch.no_grad()
def empirical_inference_flops(model, input_ids: torch.Tensor,
                              num_steps: Optional[int] = None) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with counter:
        _forward_logits(model, input_ids, num_steps=num_steps)
    return counter.get_total_flops()


def _forward_logits(model, input_ids: torch.Tensor,
                    num_steps: Optional[int] = None):
    if hasattr(model, "forward_for_generation"):
        kwargs = {}
        if num_steps is not None:
            kwargs["num_steps"] = num_steps
        return model.forward_for_generation(input_ids, **kwargs)["logits"]
    out = model(input_ids, return_logits=True)
    return out["logits"] if isinstance(out, dict) else out


@torch.no_grad()
def latency_bench(model, input_ids: torch.Tensor, *, warmup: int, iters: int,
                  autocast_dtype: Optional[torch.dtype],
                  num_steps: Optional[int] = None) -> dict:
    device = input_ids.device
    use_cuda = device.type == "cuda"
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
                    if use_cuda and autocast_dtype is not None
                    else _NullCtx())

    for _ in range(warmup):
        with autocast_ctx:
            _ = _forward_logits(model, input_ids, num_steps=num_steps)
    if use_cuda:
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_ctx:
                _ = _forward_logits(model, input_ids, num_steps=num_steps)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            with autocast_ctx:
                _ = _forward_logits(model, input_ids, num_steps=num_steps)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_sorted = sorted(times_ms)

    def pct(p):
        idx = max(0, min(len(times_sorted) - 1, int(round(p * (len(times_sorted) - 1)))))
        return times_sorted[idx]

    return {
        "iters": iters,
        "warmup": warmup,
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
    }


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_int(n): return f"{n:,}"
def _fmt_M(n): return f"{n / 1e6:8.2f}M"
def _fmt_G(n): return f"{n / 1e9:8.3f}G"
def _fmt_T(n): return f"{n / 1e12:8.3f}T"


def _table(title, rows):
    keys = list(rows[0][1].keys())
    widths = {k: max(len(k), max(len(str(r[1].get(k, ""))) for r in rows)) for k in keys}
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    head = "  model  | " + " | ".join(f"{k:>{widths[k]}}" for k in keys)
    print(head)
    print("-" * len(head))
    for label, row in rows:
        print(f"  {label:>5} | " + " | ".join(f"{str(row.get(k, '-')):>{widths[k]}}" for k in keys))


def _eqlm_label(stats: dict) -> str:
    ns = stats.get("num_steps")
    return f"eqlm@{ns}" if ns is not None else "attractor"


def print_report(parcae_stats: dict, eqlm_runs: list[dict]):
    rows = [("parcae", {
        "params": _fmt_M(parcae_stats["params"]["total"]),
        "wte": _fmt_M(parcae_stats["params"].get("wte", 0)),
        "core_block": _fmt_M(parcae_stats["params"].get("core_block", 0)),
        "prelude": _fmt_M(parcae_stats["params"].get("prelude", 0)),
        "coda": _fmt_M(parcae_stats["params"].get("coda", 0)),
        "h(gpt)": _fmt_M(parcae_stats["params"].get("h", 0)),
    })]
    eqlm_first = eqlm_runs[0]
    rows.append(("attractor", {
        "params": _fmt_M(eqlm_first["params"]["total"]),
        "wte": _fmt_M(eqlm_first["params"].get("wte", 0)),
        "core_block": _fmt_M(eqlm_first["params"].get("core_block", 0)),
        "prelude": _fmt_M(eqlm_first["params"].get("prelude", 0)),
        "coda": _fmt_M(eqlm_first["params"].get("coda", 0)),
        "h(gpt)": _fmt_M(eqlm_first["params"].get("h", 0)),
    }))
    _table("PARAMETER COUNTS (M)", rows)

    rows = [("parcae", {
        "is_recurrent": str(parcae_stats["analytical"]["is_recurrent"]),
        "mean_rec": parcae_stats["analytical"]["mean_recurrence"],
        "bw_depth": parcae_stats["analytical"]["mean_backprop_depth"],
        "train_flops/tok": _fmt_G(parcae_stats["analytical"]["flops_per_token_train"]),
    })]
    for r in eqlm_runs:
        rows.append((_eqlm_label(r), {
            "is_recurrent": str(r["analytical"]["is_recurrent"]),
            "mean_rec": r["analytical"]["mean_recurrence"],
            "bw_depth": r["analytical"]["mean_backprop_depth"],
            "train_flops/tok": _fmt_G(r["analytical"]["flops_per_token_train"]),
        }))
    _table("ANALYTICAL TRAINING FLOPs/TOKEN  (mean_rec = num_steps used)", rows)

    p_inf = parcae_stats["inference"]
    rows = [("parcae", {
        "shape": f"{p_inf['B']}x{p_inf['T']}",
        "num_steps": p_inf.get("num_steps", "-"),
        "flops/forward": _fmt_T(p_inf["flops_total"]),
        "flops/token": _fmt_G(p_inf["flops_per_token"]),
    })]
    for r in eqlm_runs:
        e_inf = r["inference"]
        rows.append((_eqlm_label(r), {
            "shape": f"{e_inf['B']}x{e_inf['T']}",
            "num_steps": e_inf.get("num_steps", "-"),
            "flops/forward": _fmt_T(e_inf["flops_total"]),
            "flops/token": _fmt_G(e_inf["flops_per_token"]),
        }))
    _table("EMPIRICAL INFERENCE FLOPs (single forward, FlopCounterMode)", rows)

    p_lat = parcae_stats["latency"]
    rows = [("parcae", {
        "num_steps": parcae_stats.get("num_steps", "-"),
        "iters": p_lat["iters"],
        "mean_ms": f"{p_lat['mean_ms']:.2f}",
        "std_ms": f"{p_lat['std_ms']:.2f}",
        "p50_ms": f"{p_lat['p50_ms']:.2f}",
        "p95_ms": f"{p_lat['p95_ms']:.2f}",
        "tok/s": f"{parcae_stats['throughput_tok_per_s']:.1f}",
    })]
    for r in eqlm_runs:
        e_lat = r["latency"]
        rows.append((_eqlm_label(r), {
            "num_steps": r.get("num_steps", "-"),
            "iters": e_lat["iters"],
            "mean_ms": f"{e_lat['mean_ms']:.2f}",
            "std_ms": f"{e_lat['std_ms']:.2f}",
            "p50_ms": f"{e_lat['p50_ms']:.2f}",
            "p95_ms": f"{e_lat['p95_ms']:.2f}",
            "tok/s": f"{r['throughput_tok_per_s']:.1f}",
        }))
    _table("FORWARD LATENCY (cuda events, autocast=bf16)", rows)

    print()
    print("=" * 80)
    print("RATIO eqlm@<num_steps> / parcae")
    print("=" * 80)
    p_train = parcae_stats["analytical"]["flops_per_token_train"]
    p_inf_tok = parcae_stats["inference"]["flops_per_token"]
    p_ms = parcae_stats["latency"]["mean_ms"]
    p_tok = parcae_stats["throughput_tok_per_s"]
    print(f"  {'tag':<14s} {'params':>10s} {'train_f/t':>12s} {'infer_f/t':>12s} "
          f"{'latency':>10s} {'tok/s':>10s}")
    for r in eqlm_runs:
        tag = _eqlm_label(r)
        e_train = r["analytical"]["flops_per_token_train"]
        e_inf_tok = r["inference"]["flops_per_token"]
        e_ms = r["latency"]["mean_ms"]
        e_tok = r["throughput_tok_per_s"]
        print(f"  {tag:<14s} "
              f"{r['params']['total'] / max(parcae_stats['params']['total'], 1):>9.3f}x "
              f"{e_train / max(p_train, 1):>11.3f}x "
              f"{e_inf_tok / max(p_inf_tok, 1):>11.3f}x "
              f"{e_ms / max(p_ms, 1e-6):>9.3f}x "
              f"{e_tok / max(p_tok, 1e-6):>9.3f}x")

    print()
    print("EQLM forward-solver diagnostics per measurement")
    print("-" * 76)
    print(f"  {'tag':<14s} {'calls':>6s} {'mean_iters':>12s} {'mean_rel':>12s} "
          f"{'frac_conv':>10s} {'max_iters':>10s}")
    for r in eqlm_runs:
        s = r.get("solver_summary")
        if s is None:
            continue
        print(f"  {_eqlm_label(r):<14s} "
              f"{s['calls']:>6d} "
              f"{s['mean_iters']:>12.2f} "
              f"{s['mean_rel_res']:>12.2e} "
              f"{s['frac_converged']:>10.3f} "
              f"{int(s['max_iters_seen']):>10d}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_generate(model, prompt: torch.Tensor, gen_tokens: int,
                  num_steps: Optional[int], use_cache: bool,
                  autocast_dtype: Optional[torch.dtype]) -> dict:
    import inspect
    if not hasattr(model, "generate"):
        return {"elapsed_ms": float("nan"), "new_tokens": 0,
                "tok_per_s": 0.0, "ms_per_tok": float("nan"),
                "error": f"{type(model).__name__} has no generate method"}
    use_cuda = prompt.device.type == "cuda"
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
                    if use_cuda and autocast_dtype is not None else _NullCtx())
    sig = inspect.signature(model.generate).parameters
    kwargs = dict(max_new_tokens=gen_tokens, do_sample=False, temperature=1.0)
    if "num_steps" in sig and num_steps is not None:
        kwargs["num_steps"] = num_steps
    if "use_cache" in sig:
        kwargs["use_cache"] = use_cache

    # warmup
    with autocast_ctx:
        _ = model.generate(prompt, **{**kwargs, "max_new_tokens": min(4, gen_tokens)})
    if use_cuda:
        torch.cuda.synchronize()

    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with autocast_ctx:
            out = model.generate(prompt, **kwargs)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
    else:
        t0 = time.time()
        with autocast_ctx:
            out = model.generate(prompt, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000.0

    new_tokens = out.shape[1] - prompt.shape[1]
    return {
        "elapsed_ms": elapsed_ms,
        "new_tokens": new_tokens,
        "tok_per_s": (new_tokens * 1000.0) / max(elapsed_ms, 1e-6),
        "ms_per_tok": elapsed_ms / max(new_tokens, 1),
    }


def run_generation_bench(parcae_model, eqlm_model, eqlm_targets, *,
                         device: str, vocab_size: int, prompt_len: int,
                         gen_tokens: int,
                         autocast_dtype: Optional[torch.dtype],
                         use_cache: bool) -> dict:
    torch.manual_seed(0)
    prompt = torch.randint(0, max(2, vocab_size), (1, prompt_len), device=device)
    print(f"\nMeasuring generation: prompt_len={prompt_len}  "
          f"gen_tokens={gen_tokens}  use_cache={use_cache}")
    parcae_g = _run_generate(parcae_model, prompt, gen_tokens,
                             num_steps=None, use_cache=use_cache,
                             autocast_dtype=autocast_dtype)
    eqlm_runs = []
    for ns in eqlm_targets:
        if ns is not None:
            eqlm_model.apply_eval_solver(max_iter=int(ns))
        if hasattr(eqlm_model, "reset_solver_stats"):
            eqlm_model.reset_solver_stats()
        g = _run_generate(eqlm_model, prompt, gen_tokens,
                          num_steps=ns, use_cache=use_cache,
                          autocast_dtype=autocast_dtype)
        if hasattr(eqlm_model, "solver_summary"):
            g["solver_summary"] = eqlm_model.solver_summary()
        g["num_steps"] = ns
        eqlm_runs.append(g)
    return {"parcae": parcae_g, "attractor": eqlm_runs,
            "prompt_len": prompt_len, "gen_tokens": gen_tokens}


def print_generation_report(gen: dict, *, use_cache: bool) -> None:
    print()
    print("=" * 80)
    label = "with KV cache" if use_cache else "no KV cache"
    print(f"AUTOREGRESSIVE GENERATION  (greedy, prompt={gen['prompt_len']}, "
          f"+{gen['gen_tokens']} tokens, {label})")
    print("=" * 80)

    def _row(d):
        if d.get("error"):
            return {"ms_total": "-", "ms/tok": "-", "tok/s": f"skip ({d['error']})"}
        return {
            "ms_total": f"{d['elapsed_ms']:.1f}",
            "ms/tok": f"{d['ms_per_tok']:.2f}",
            "tok/s": f"{d['tok_per_s']:.1f}",
        }

    rows = [("parcae", _row(gen["parcae"]))]
    for r in gen["attractor"]:
        tag = f"eqlm@{r['num_steps']}" if r.get("num_steps") is not None else "attractor"
        rows.append((tag, _row(r)))
    _table("(timings)", rows)


def measure_one(model, *, device: str, B: int, T: int, vocab_size: int,
                warmup: int, iters: int,
                autocast_dtype: Optional[torch.dtype],
                num_steps: Optional[int] = None) -> dict:
    torch.manual_seed(0)
    input_ids = torch.randint(0, max(2, vocab_size), (B, T), device=device)

    inf_flops = empirical_inference_flops(model, input_ids, num_steps=num_steps)

    if hasattr(model, "reset_solver_stats"):
        model.reset_solver_stats()

    lat = latency_bench(model, input_ids,
                        warmup=warmup, iters=iters,
                        autocast_dtype=autocast_dtype,
                        num_steps=num_steps)

    out = {
        "params": param_counts(model),
        "analytical": analytical_flops(model, num_steps=num_steps),
        "inference": {
            "B": B, "T": T,
            "flops_total": inf_flops,
            "flops_per_token": inf_flops / max(B * T, 1),
            "num_steps": num_steps,
        },
        "latency": lat,
        "throughput_tok_per_s": (B * T * 1000.0) / max(lat["mean_ms"], 1e-6),
        "num_steps": num_steps,
    }
    if hasattr(model, "solver_summary"):
        out["solver_summary"] = model.solver_summary()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parcae_repo", default="SandyResearch/parcae-140m")
    p.add_argument("--eqlm_dir", default="/scratch1/feinashl/eqlm/eqlm-small-140m")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--num_steps", type=int, default=None,
                   help="Cap recurrence depth for both models. "
                        "Parcae: forwarded as num_steps; "
                        "EQLM: caps the FP solver max_iter for that call.")
    p.add_argument("--sweep_max_iter", type=str, default=None,
                   help="Comma-separated EQLM max_iter values to sweep "
                        "(e.g. '8,16,32,64'). One Parcae measurement, "
                        "one EQLM measurement per value.")
    p.add_argument("--eval_solver", type=str, default=None,
                   help="JSON dict applied to EQLM via apply_eval_solver "
                        "before measurement (e.g. '{\"tol\": 1e-4, "
                        "\"min_iter\": 8}'). max_iter is overridden by "
                        "--sweep_max_iter when sweeping.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile EQLM prelude + core blocks. "
                        "Parcae is left eager (its ParcaeDynamicCache uses "
                        "int(step_idx_tensor) which graph-breaks during "
                        "generation and hangs).")
    p.add_argument("--compile_parcae", action="store_true",
                   help="Also torch.compile Parcae blocks. Off by default; "
                        "currently incompatible with Parcae's generation path.")
    p.add_argument("--fpi", action="store_true",
                   help="EQLM only: switch the forward solver to FPI at eval "
                        "time (shortcut for --eval_solver '{\"solver\":\"fpi\"}')")
    p.add_argument("--gen_tokens", type=int, default=0,
                   help="If > 0, also run an autoregressive generation "
                        "benchmark for both models (B=1, T=seq_len prompt, "
                        "N=gen_tokens new tokens). Reports tok/s.")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable KV cache during the generation benchmark "
                        "(useful to A/B the cache speedup).")
    p.add_argument("--out", default=None,
                   help="Save the full results JSON here. "
                        "Default: logs/compare/<timestamp>.json")
    args = p.parse_args()

    eval_solver = json.loads(args.eval_solver) if args.eval_solver else {}
    if args.fpi:
        eval_solver.setdefault("solver", "fpi")
    sweep = ([int(x) for x in args.sweep_max_iter.split(",") if x.strip()]
             if args.sweep_max_iter else None)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU. "
              "Latency numbers will be meaningless.")
        device = "cpu"
        param_dtype = torch.float32
        autocast_dtype = None
    else:
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        param_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[args.precision]
        autocast_dtype = (torch.bfloat16 if args.precision == "bf16"
                          else torch.float16 if args.precision == "fp16"
                          else None)

    print("=" * 80)
    print("MODEL COMPARISON  (FLOPs + latency)")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"precision    : {args.precision}")
    print(f"shape        : B={args.batch_size}  T={args.seq_len}")
    print(f"warmup/iters : {args.warmup}/{args.iters}")
    print(f"parcae repo  : {args.parcae_repo}")
    print(f"eqlm dir     : {args.eqlm_dir}")
    print(f"num_steps    : {args.num_steps}")
    print(f"sweep_max_it : {sweep}")
    print(f"eval_solver  : {eval_solver}")
    print("=" * 80)

    parcae_model = load_parcae_hf(args.parcae_repo, device=device, dtype=param_dtype)
    eqlm_model = load_eqlm_run(args.eqlm_dir, device=device, dtype=param_dtype)

    if eval_solver:
        eval_solver_for_eqlm = {k: v for k, v in eval_solver.items()
                                if not (sweep and k == "max_iter")}
        if eval_solver_for_eqlm:
            print(f"\nApplying eval_solver to EQLM: {eval_solver_for_eqlm}")
            eqlm_model.apply_eval_solver(**eval_solver_for_eqlm)

    if args.compile:
        print("\ntorch.compile EQLM blocks ...")
        eqlm_model.compile_for_inference()
        print("torch.compile EQLM done (first forward will trace; warmup absorbs it).")
    if args.compile_parcae:
        print("\ntorch.compile Parcae blocks (may stall generation due to "
              "ParcaeDynamicCache graph-breaks).")
        if hasattr(parcae_model, "transformer"):
            tr = parcae_model.transformer
            for attr in ("prelude", "core_block", "coda", "h"):
                if hasattr(tr, attr):
                    setattr(tr, attr, torch.nn.ModuleList([
                        torch.compile(b, mode="default", dynamic=True)
                        for b in getattr(tr, attr)
                    ]))

    parcae_vocab = int(parcae_model.config.padded_vocab_size)
    eqlm_vocab = int(eqlm_model.config.padded_vocab_size)
    vocab_size = min(parcae_vocab, eqlm_vocab)

    print(f"\nMeasuring parcae (num_steps={args.num_steps}) ...")
    parcae_stats = measure_one(parcae_model, device=device,
                               B=args.batch_size, T=args.seq_len,
                               vocab_size=vocab_size,
                               warmup=args.warmup, iters=args.iters,
                               autocast_dtype=autocast_dtype,
                               num_steps=args.num_steps)

    eqlm_runs = []
    eqlm_targets = sweep if sweep else [args.num_steps]
    for ns in eqlm_targets:
        print(f"Measuring eqlm  (num_steps={ns}) ...")
        if ns is not None:
            eqlm_model.apply_eval_solver(max_iter=int(ns))
        stats = measure_one(eqlm_model, device=device,
                            B=args.batch_size, T=args.seq_len,
                            vocab_size=vocab_size,
                            warmup=args.warmup, iters=args.iters,
                            autocast_dtype=autocast_dtype,
                            num_steps=ns)
        eqlm_runs.append(stats)

    gen_results = None
    if args.gen_tokens > 0:
        gen_results = run_generation_bench(
            parcae_model, eqlm_model, eqlm_targets,
            device=device, vocab_size=vocab_size,
            prompt_len=args.seq_len, gen_tokens=args.gen_tokens,
            autocast_dtype=autocast_dtype,
            use_cache=not args.no_cache)

    print_report(parcae_stats, eqlm_runs)
    if gen_results is not None:
        print_generation_report(gen_results, use_cache=not args.no_cache)

    out_path = (Path(args.out) if args.out
                else Path("logs/compare") / f"compare_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "device": device,
        "vocab_size_used": vocab_size,
        "eval_solver_applied": eval_solver,
        "parcae": parcae_stats,
        "eqlm_runs": eqlm_runs,
        "generation": gen_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved JSON -> {out_path}")


if __name__ == "__main__":
    main()
