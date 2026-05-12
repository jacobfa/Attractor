"""Extract just the model weights from a parcae training-state checkpoint and
save in HF-compatible format (config.json + model.pt) using
``attractor.save_pretrained``.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attractor
from attractor.models.attractor.config import AttractorConfig


def _strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        for p in ("module.", "_orig_mod.", "model.", "_forward_module."):
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Parcae training run dir (contains model_config.json + checkpoints-DDPStrategy/)")
    ap.add_argument("--ckpt_glob", default="last-*",
                    help="Glob inside checkpoints-DDPStrategy/ to pick the checkpoint (default: last-*)")
    ap.add_argument("--out_dir", required=True,
                    help="Destination directory to write HF-compatible files into")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "model_config.json"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    print(f"[load] model_config.json: name={cfg_dict.get('name')}  arch={cfg_dict.get('model_class_name', cfg_dict.get('architecture_class_name'))}")

    cfg_dict.pop("rope_settings", None)  # rebuilt by post_init from defaults
    config = AttractorConfig(**{k: v for k, v in cfg_dict.items()
                           if k in AttractorConfig.__dataclass_fields__})
    model = config.construct_model()

    ckpt_dir = run_dir / "checkpoints-DDPStrategy"
    matches = sorted(ckpt_dir.glob(args.ckpt_glob))
    if not matches:
        raise FileNotFoundError(f"no ckpt match {ckpt_dir}/{args.ckpt_glob}")
    ckpt_path = matches[0]
    print(f"[load] checkpoint: {ckpt_path}  ({ckpt_path.stat().st_size/1e9:.2f} GB)")

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw)
    sd = _strip_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}  (first 3: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}  (first 3: {unexpected[:3]})")

    attractor.save_pretrained(model, str(out_dir))
    written = sorted(p.name for p in out_dir.iterdir())
    print(f"[done] {out_dir}: {written}")
    for p in out_dir.iterdir():
        print(f"        {p.name:30s}  {p.stat().st_size/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
