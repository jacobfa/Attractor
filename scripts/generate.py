"""Standalone text generation for GPT / Parcae / EQLM checkpoints."""
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from jsonargparse import CLI

sys.path.insert(0, str(Path(__file__).parent.parent))

from receval.settings import CLISettings as EvalSettings


@dataclass
class GenerateSettings:
    out_dir: str = ""
    checkpoint_path: Optional[str] = None
    hf_path: Optional[str] = None
    hf_repo: Optional[str] = None
    step: Optional[int] = None
    prompt: str = "The meaning of life is"
    prompts_file: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    do_sample: bool = True
    seed: int = 42
    device_type: str = ""
    precision: str = "bf16"
    num_steps: Optional[int] = None
    eval_solver: dict = field(default_factory=dict)
    save_path: Optional[str] = None


def _build_eval_settings(g: GenerateSettings) -> EvalSettings:
    return EvalSettings(
        out_dir=g.out_dir,
        eval_name="generate",
        eval_tasks="sample",
        step=g.step,
        checkpoint_path=g.checkpoint_path,
        hf_path=g.hf_path,
        hf_repo=g.hf_repo,
        device_type=g.device_type,
        precision=g.precision,
        eval_solver=g.eval_solver,
    )


def _load_model(s: EvalSettings):
    from attractor.tokenizer import Tokenizer
    if s.checkpoint_path is None and s.hf_path is None and s.hf_repo is None:
        raise SystemExit(
            "No checkpoint found. Pass --out_dir <run_dir>, "
            "--checkpoint_path <path>, --hf_repo <id>, or --hf_path <id>.")

    if s.hf_path is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(s.hf_path).to(s.device).eval()
        return model, AutoTokenizer.from_pretrained(s.hf_path)

    if s.hf_repo is not None:
        if s.model_impl in ("attractor", "eqlm"):
            from receval.models.attractor import ModelingAttractor as Cls
        elif s.model_impl == "parcae":
            from receval.models.parcae import ModelingParcae as Cls
        else:
            from receval.models.gpt import ModelingGPT as Cls
        model = Cls.from_pretrained(s.hf_repo, device=s.device)
        tok = (Tokenizer(s.tokenizer_path) if s.tokenizer_path
               else Tokenizer.from_pretrained("SandyResearch/parcae-tokenizer"))
        return model, tok

    if s.model_impl in ("attractor", "eqlm"):
        from receval.models.attractor import ModelingAttractor as Cls
    elif s.model_impl == "parcae":
        from receval.models.parcae import ModelingParcae as Cls
    else:
        from receval.models.gpt import ModelingGPT as Cls
    assert s.model_config is not None, "model_config required"
    model = Cls(s.model_config)
    state = torch.load(s.checkpoint_path, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    cleaned = {}
    for k, v in sd.items():
        for prefix in ["module.", "_orig_mod.", "model.", "_forward_module."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(s.device).eval()
    if s.model_impl in ("attractor", "eqlm") and s.eval_solver:
        print(f"Applying eval_solver overrides: {s.eval_solver}")
        model.apply_eval_solver(**s.eval_solver)
    if not s.tokenizer_path:
        raise ValueError("tokenizer_path not found in run_config.json")
    return model, Tokenizer(s.tokenizer_path)


def _encode_prompt(tokenizer, text: str) -> list[int]:
    bos = getattr(tokenizer, "bos_id", None) or getattr(tokenizer, "bos_token_id", None)
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if not isinstance(ids, list):
        ids = list(ids)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


def _save_generations(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for prompt, output in rows:
            f.write(json.dumps({"prompt": prompt, "output": output}) + "\n")


def main():
    g: GenerateSettings = CLI(GenerateSettings)
    torch.manual_seed(g.seed)

    s = _build_eval_settings(g)
    print("=" * 60)
    print("Generation")
    print("=" * 60)
    print(f"Run dir   : {s.out_dir}")
    print(f"Model     : {s.model_name}  (impl={s.model_impl})")
    print(f"Checkpoint: {s.checkpoint_path}")
    print(f"Device    : {s.device}")
    print("=" * 60)

    model, tokenizer = _load_model(s)

    if g.prompts_file:
        with open(g.prompts_file) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        prompts = [g.prompt]

    rows = []
    for i, prompt in enumerate(prompts):
        ids = _encode_prompt(tokenizer, prompt)
        ctx = torch.tensor(ids, dtype=torch.long, device=s.device).unsqueeze(0)
        t0 = time.time()
        with s._get_autocast_context():
            gen_kwargs = dict(
                max_new_tokens=g.max_new_tokens,
                temperature=g.temperature,
                top_k=g.top_k,
                top_p=g.top_p,
                do_sample=g.do_sample,
            )
            if g.num_steps is not None and "num_steps" in model.generate.__code__.co_varnames:
                gen_kwargs["num_steps"] = g.num_steps
            out = model.generate(ctx, **gen_kwargs)
        dt = time.time() - t0
        new_tokens = out[0, len(ids):].tolist()
        text = tokenizer.decode(new_tokens, skip_special_tokens=True) \
            if "skip_special_tokens" in tokenizer.decode.__code__.co_varnames \
            else tokenizer.decode(new_tokens)
        full = prompt + text
        rows.append((prompt, full))
        print(f"\n[{i + 1}/{len(prompts)}]  {len(new_tokens)} tok in {dt:.2f}s "
              f"({len(new_tokens) / max(dt, 1e-6):.1f} tok/s)")
        print(f"[Prompt] {prompt}")
        print(f"[Output] {full}")

    if s.model_impl in ("attractor", "eqlm") and hasattr(model, "solver_summary"):
        stats = model.solver_summary()
        if stats["calls"] > 0:
            print("\n" + "-" * 60)
            print("  EQLM forward-solver diagnostics during generation")
            print("-" * 60)
            print(f"    forward calls       : {stats['calls']:>10d}")
            print(f"    mean iters / sample : {stats['mean_iters']:>10.2f}")
            print(f"    mean rel_residual   : {stats['mean_rel_res']:>10.2e}")
            print(f"    fraction converged  : {stats['frac_converged']:>10.3f}")
            print(f"    max iters seen      : {stats['max_iters_seen']:>10.1f}")
            print("-" * 60)

    save_path = (Path(g.save_path) if g.save_path
                 else Path(s.out_dir) / "eval" / "generations.jsonl")
    _save_generations(rows, save_path)
    print(f"\nSaved {len(rows)} generation(s) -> {save_path}")


if __name__ == "__main__":
    main()
