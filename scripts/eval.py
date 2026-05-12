import os
import sys
import time
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jsonargparse import CLI
from receval.settings import CLISettings
from receval.tasks import run_lm_eval
from receval.tasks.val_loss import run_val_loss, load_val_texts_from_parquet
from receval.tasks.core_eval import run_core_eval
from receval.tasks.core_extended_eval import run_core_extended_eval



def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs, flush=True)


def setup_distributed(settings: CLISettings):
    if settings.ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(settings.ddp_local_rank)
    if settings.device_type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except AttributeError:
            pass  # Not all PyTorch versions have this


def cleanup_distributed(settings: CLISettings):
    if settings.ddp:
        torch.distributed.destroy_process_group()


def _detect_hf_arch(repo_id: str) -> str:
    """Peek at config.json in a HuggingFace repo to determine the architecture."""
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    arch = cfg.get("architecture_class_name", "")
    if arch in ("EQLM", "Attractor") or "attractor" in repo_id.lower() or "eqlm" in repo_id.lower():
        return "attractor"
    if arch == "Parcae":
        return "parcae"
    return "gpt"


def load_model(settings: CLISettings):
    from attractor.tokenizer import Tokenizer

    # HuggingFace transformers model (e.g. GPT-2)
    if settings.hf_path is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print0(f"Loading HuggingFace model: {settings.hf_path}")
        model = AutoModelForCausalLM.from_pretrained(settings.hf_path)
        if settings.device is not None:
            model.to(settings.device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(settings.hf_path)
        return model, tokenizer

    # Parcae/GPT/EQLM model from HuggingFace repo
    if settings.hf_repo is not None:
        arch = _detect_hf_arch(settings.hf_repo)
        print0(f"Loading from HuggingFace: {settings.hf_repo} (detected arch={arch})")
        if arch in ("attractor", "eqlm"):
            from receval.models.attractor import ModelingAttractor
            model = ModelingEQLM.from_pretrained(settings.hf_repo, device=settings.device)
            settings.model_impl = "attractor"
            if settings.eval_solver:
                print0(f"Applying eval_solver overrides: {settings.eval_solver}")
                model.apply_eval_solver(**settings.eval_solver)
        else:
            from receval.models.parcae import ModelingParcae
            model = ModelingParcae.from_pretrained(settings.hf_repo, device=settings.device)
        model.eval()
        if settings.tokenizer_path:
            tokenizer = Tokenizer(settings.tokenizer_path)
        elif arch in ("attractor", "eqlm"):
            # EQLM HF repos (jacobfa1/*) were trained with the local parcae
            # tokenizer; pin to it so token IDs match training.
            local_tok = "/scratch1/feinashl/tokenizer/parcae_tokenizer"
            from pathlib import Path as _P
            if _P(local_tok).exists():
                print0(f"Using local training tokenizer for EQLM: {local_tok}")
                tokenizer = Tokenizer(local_tok)
            else:
                tokenizer = Tokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
        else:
            tokenizer = Tokenizer.from_pretrained("SandyResearch/parcae-tokenizer")
        return model, tokenizer

    # Local checkpoint
    if settings.model_impl == "gpt":
        from receval.models.gpt import ModelingGPT
        ModelClass = ModelingGPT
    elif settings.model_impl == "parcae":
        from receval.models.parcae import ModelingParcae
        ModelClass = ModelingParcae
    elif settings.model_impl in ("attractor", "eqlm"):
        from receval.models.attractor import ModelingAttractor
        ModelClass = ModelingEQLM
    else:
        raise ValueError(f"Unknown model_impl: {settings.model_impl}")

    print0(f"Loading {settings.model_impl} model: {settings.model_name}")
    assert settings.model_config is not None, "model_config required"

    if settings.override_mean_recurrence is not None and hasattr(settings.model_config, 'mean_recurrence'):
        original_recurrence = settings.model_config.mean_recurrence
        settings.model_config.mean_recurrence = settings.override_mean_recurrence
        print0(f"Overriding mean_recurrence: {original_recurrence} -> {settings.override_mean_recurrence}")

    model = ModelClass(settings.model_config)

    if settings.checkpoint_path:
        print0(f"Loading checkpoint: {settings.checkpoint_path}")
        state = torch.load(settings.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.to(settings.device)
    model.eval()

    if (settings.override_mean_recurrence is not None
            and hasattr(model, "config")
            and hasattr(model.config, "mean_recurrence")):
        prev = model.config.mean_recurrence
        model.config.mean_recurrence = int(settings.override_mean_recurrence)
        print0(f"Overriding model.config.mean_recurrence: {prev} -> "
               f"{model.config.mean_recurrence}")

    if settings.model_impl in ("attractor", "eqlm") and settings.eval_solver:
        print0(f"Applying eval_solver overrides: {settings.eval_solver}")
        model.apply_eval_solver(**settings.eval_solver)

    if settings.tokenizer_path:
        tokenizer = Tokenizer(settings.tokenizer_path)
    else:
        raise ValueError("tokenizer_path not found in run config")

    return model, tokenizer


def run_sample_task(model, tokenizer, settings: CLISettings):
    if not settings._is_main_process:
        return {}

    task_settings = settings.tasks.sample
    results = {"conditioned": [], "unconditioned": []}

    print0("\n" + "=" * 60)
    print0("Sample Generation")
    print0("=" * 60)

    for prompt in task_settings.prompts:
        tokens = tokenizer.encode(prompt, return_tensors=False)
        input_ids = torch.tensor([tokens], device=settings.device)
        with settings._get_autocast_context():
            output = model.generate(
                input_ids,
                max_new_tokens=task_settings.max_tokens,
                temperature=task_settings.temperature,
                top_k=task_settings.top_k,
                top_p=task_settings.top_p,
                do_sample=task_settings.temperature > 0,
            )
        text = tokenizer.decode(output[0].tolist())
        print0(f"\n{text}")
        results["conditioned"].append(text)

    if task_settings.num_unconditioned > 0:
        print0("\nUnconditioned samples:")
        bos = getattr(tokenizer, "bos_id", 1)
        input_ids = torch.tensor([[bos]], device=settings.device)
        for _ in range(task_settings.num_unconditioned):
            with settings._get_autocast_context():
                output = model.generate(
                    input_ids,
                    max_new_tokens=task_settings.max_tokens,
                    temperature=task_settings.temperature,
                    do_sample=True,
                )
            text = tokenizer.decode(output[0].tolist())
            print0(f"\n{text}")
            results["unconditioned"].append(text)

    return results


def main():
    settings: CLISettings = CLI(CLISettings)
    setup_distributed(settings)

    print0("=" * 60)
    print0("Evaluation")
    print0("=" * 60)
    print0(f"Run dir: {settings.out_dir}")
    print0(f"Model: {settings.model_name}")
    print0(f"Checkpoint: {settings.checkpoint_path}")
    print0(f"Tasks: {settings.eval_tasks}")
    print0(f"Device: {settings.device}")
    if settings.override_mean_recurrence is not None:
        print0(f"Override recurrence: {settings.override_mean_recurrence}")
    print0("=" * 60)

    if settings.checkpoint_path is None and settings.hf_path is None and settings.hf_repo is None:
        print0("\nERROR: No checkpoint found!")
        print0("Check that out_dir contains checkpoints or pass --checkpoint_path")
        cleanup_distributed(settings)
        return

    model, tokenizer = load_model(settings)
    all_results = {}

    is_eqlm = (settings.model_impl in ("attractor", "eqlm") and hasattr(model, "solver_summary"))

    def _snapshot():
        return dict(model._solver_stats) if is_eqlm else None

    def _delta(before, after):
        out = {k: after[k] - before[k] for k in
               ("samples", "calls", "iters_sum", "rel_res_sum",
                "converged_sum", "non_converged_calls")}
        out["max_iters_seen"] = after["max_iters_seen"]
        out["max_rel_res_seen"] = after["max_rel_res_seen"]
        n = max(out["samples"], 1)
        return {
            "samples": out["samples"], "calls": out["calls"],
            "mean_iters": out["iters_sum"] / n,
            "mean_rel_res": out["rel_res_sum"] / n,
            "frac_converged": out["converged_sum"] / n,
            "max_iters_seen": out["max_iters_seen"],
            "max_rel_res_seen": out["max_rel_res_seen"],
            "non_converged_calls": out["non_converged_calls"],
        }

    for task in settings.eval_task_list:
        print0(f"\nRunning task: {task}")
        t0 = time.time()
        snap_before = _snapshot()

        if task == "lm_eval":
            results = run_lm_eval(model, tokenizer, settings)
            all_results["lm_eval"] = results.results
            if settings._is_main_process:
                print0(f"\nLM-Eval Results:")
                for task_name, task_result in results.task_scores.items():
                    print0(f"  {task_name}:")
                    if isinstance(task_result, dict):
                        for metric, value in task_result.items():
                            if isinstance(value, (int, float)):
                                print0(f"    {metric}: {value:.4f}")
                    else:
                        print0(f"    {task_result}")

        elif task == "sample":
            results = run_sample_task(model, tokenizer, settings)
            all_results["sample"] = results

        elif task == "bpb":
            val_data_dir = settings.tasks.bpb.val_data_dir
            if not val_data_dir:
                print0("WARNING: No val_data_dir specified for BPB task, skipping")
                all_results["bpb"] = {}
            else:
                print0(f"Loading validation data from: {val_data_dir}")
                val_texts = load_val_texts_from_parquet(
                    val_data_dir,
                    max_files=settings.tasks.bpb.max_files
                )
                print0(f"Loaded {len(val_texts)} validation texts")
                results = run_val_loss(
                    model, tokenizer, settings, val_texts,
                    max_samples=settings.tasks.bpb.max_samples
                )
                all_results["bpb"] = {
                    "loss": results.loss,
                    "perplexity": results.perplexity,
                    "bits_per_byte": results.bits_per_byte,
                    "num_tokens": results.num_tokens,
                    "num_bytes": results.num_bytes,
                }
                if settings._is_main_process:
                    print0(f"\nValidation Results:")
                    print0(f"  Loss: {results.loss:.4f}")
                    print0(f"  Perplexity: {results.perplexity:.2f}")
                    print0(f"  Bits per byte: {results.bits_per_byte:.4f}")
                    print0(f"  Tokens evaluated: {results.num_tokens:,}")
                    print0(f"  Bytes evaluated: {results.num_bytes:,}")

        elif task == "core":
            max_seq_len = settings.sequence_length or 2048
            seeds = settings.tasks.core.seeds
            with settings._get_autocast_context():
                results = run_core_eval(
                    model, tokenizer, settings.device,
                    max_seq_len=max_seq_len,
                    max_per_task=settings.tasks.core.max_per_task,
                    seeds=seeds
                )
            all_results["core"] = results
            if settings._is_main_process:
                print0(f"\nCORE Results:")
                # Check if we have aggregated results (multiple seeds)
                if 'aggregated' in results:
                    for task_name in results["results"].keys():
                        acc = results["aggregated"]["results"][task_name]
                        acc_std = results["aggregated"]["results_std"][task_name]
                        centered = results["aggregated"]["centered_results"][task_name]
                        centered_std = results["aggregated"]["centered_results_std"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f}±{acc_std:.4f} centered={centered:.4f}±{centered_std:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f} ± {results['aggregated']['core_metric_std']:.4f}")
                else:
                    for task_name, acc in results["results"].items():
                        centered = results["centered_results"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f} centered={centered:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f}")

        elif task == "core_extended":
            max_seq_len = settings.sequence_length or 2048
            seeds = settings.tasks.core_extended.seeds
            with settings._get_autocast_context():
                results = run_core_extended_eval(
                    model, tokenizer, settings.device,
                    max_seq_len=max_seq_len,
                    max_per_task=settings.tasks.core_extended.max_per_task,
                    seeds=seeds
                )
            all_results["core_extended"] = results
            if settings._is_main_process:
                print0(f"\nCORE Extended Results:")
                if 'aggregated' in results:
                    for task_name in results["results"].keys():
                        acc = results["aggregated"]["results"][task_name]
                        acc_std = results["aggregated"]["results_std"][task_name]
                        centered = results["aggregated"]["centered_results"][task_name]
                        centered_std = results["aggregated"]["centered_results_std"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f}±{acc_std:.4f} centered={centered:.4f}±{centered_std:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f} ± {results['aggregated']['core_metric_std']:.4f}")
                    print0(f"  CORE Extended metric: {results['core_extended_metric']:.4f} ± {results['aggregated']['core_extended_metric_std']:.4f}")
                else:
                    for task_name, acc in results["results"].items():
                        centered = results["centered_results"][task_name]
                        print0(f"  {task_name}: acc={acc:.4f} centered={centered:.4f}")
                    print0(f"\n  CORE metric: {results['core_metric']:.4f}")
                    print0(f"  CORE Extended metric: {results['core_extended_metric']:.4f}")

        if is_eqlm and snap_before is not None:
            d = _delta(snap_before, model._solver_stats)
            if d["calls"] > 0:
                all_results.setdefault("eqlm_solver_per_task", {})[task] = d
                print0(f"  [eqlm] {task}: mean_iters={d['mean_iters']:.2f} "
                       f"mean_rel_res={d['mean_rel_res']:.2e} "
                       f"frac_conv={d['frac_converged']:.3f} "
                       f"max_iters={int(d['max_iters_seen'])} "
                       f"calls={d['calls']} samples={d['samples']}")
        print0(f"Task {task} completed in {time.time() - t0:.2f}s")

    if settings.model_impl in ("attractor", "eqlm") and hasattr(model, "solver_summary"):
        stats = model.solver_summary()
        if stats["calls"] > 0:
            all_results["eqlm_solver"] = stats
            if settings._is_main_process:
                print0("")
                print0("-" * 60)
                print0("  EQLM forward-solver diagnostics over full eval")
                print0("-" * 60)
                print0(f"    samples scored            : {stats['samples']:>10d}")
                print0(f"    forward calls             : {stats['calls']:>10d}")
                print0(f"    mean iters / sample       : {stats['mean_iters']:>10.2f}")
                print0(f"    mean rel_residual         : {stats['mean_rel_res']:>10.2e}")
                print0(f"    fraction converged        : {stats['frac_converged']:>10.3f}")
                print0(f"    max iters seen            : {stats['max_iters_seen']:>10.1f}")
                print0(f"    max rel_res seen          : {stats['max_rel_res_seen']:>10.2e}")
                print0(f"    calls w/ non-converged    : {stats['non_converged_calls']:>10d} "
                       f"({100*stats['frac_calls_with_nonconv']:.1f}%)")
                print0("-" * 60)

    if settings._is_main_process and settings.out_dir:
        eval_dir = Path(settings.out_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        output_path = eval_dir / f"{settings.eval_name}.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print0(f"\nResults saved to: {output_path}")

    cleanup_distributed(settings)
    print0("\nEvaluation complete.")


if __name__ == "__main__":
    main()
