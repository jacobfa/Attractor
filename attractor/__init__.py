def create_config(name: str, **kwargs):
    """Create a model config by name. Returns a GPTConfig or ParcaeConfig.

    >>> config = create_config("parcae-small-140m")
    >>> config.block_size = 2048
    >>> model = config.construct_model()
    """
    from attractor.models.config import Config
    return Config.from_name(name, **kwargs)


def create_model(name: str, **kwargs):
    """Create a model by name, ready for forward passes.

    >>> model = create_model("parcae-small-140m")
    >>> model = create_model("gpt-small-140m")
    """
    config = create_config(name, **kwargs)
    return config.construct_model()


def from_pretrained(repo_id: str, device="cpu", dtype=None, **kwargs):
    """Load a pretrained model from a HuggingFace repository.

    Expects the repo to contain:
      - config.json: serialized model config (with _class_name field)
      - pytorch_model.bin or model.pt: PyTorch state dict

    >>> model = attractor.from_pretrained("SandyResearch/parcae-140m")
    >>> model = attractor.from_pretrained("SandyResearch/parcae-140m", device="cuda")
    """
    import json
    import torch
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id, "config.json", **kwargs)
    with open(config_path) as f:
        config_dict = json.load(f)

    class_name = config_dict.pop("_class_name", "GPTConfig")
    config_dict.pop("rope_settings", None)

    # Lazy import to avoid circular deps — Config.from_name handles dispatch internally
    if class_name == "ParcaeConfig":
        from attractor.models.parcae.config import ParcaeConfig as config_cls
    elif class_name == "EQLMConfig":
        from attractor.models.eqlm.config import EQLMConfig as config_cls
    else:
        from attractor.models.gpt.config import GPTConfig as config_cls

    config = config_cls(**config_dict)

    model = config.construct_model()

    for weight_file in ["pytorch_model.bin", "model.pt"]:
        try:
            weights_path = hf_hub_download(repo_id, weight_file, **kwargs)
            break
        except Exception:
            continue
    else:
        raise FileNotFoundError(f"No weights found in {repo_id} (tried pytorch_model.bin, model.pt)")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)

    return model


def save_pretrained(model, path: str):
    """Save a model's config and weights for later loading with from_pretrained.

    >>> attractor.save_pretrained(model, "./my-model")
    # Then upload the directory to HuggingFace
    """
    import json
    import torch
    from pathlib import Path
    from dataclasses import asdict

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(model.config)
    config_dict["_class_name"] = model.config.__class__.__name__
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    torch.save(model.state_dict(), out_dir / "model.pt")


def __getattr__(name):
    if name == "GPT":
        from attractor.models.gpt.gpt import GPT
        return GPT
    if name == "GPTConfig":
        from attractor.models.gpt.config import GPTConfig
        return GPTConfig
    if name == "Parcae":
        from attractor.models.parcae.parcae import Parcae
        return Parcae
    if name == "ParcaeConfig":
        from attractor.models.parcae.config import ParcaeConfig
        return ParcaeConfig
    if name == "EQLM":
        from attractor.models.eqlm.eqlm import EQLM
        return EQLM
    if name == "EQLMConfig":
        from attractor.models.eqlm.config import EQLMConfig
        return EQLMConfig
    raise AttributeError(f"module 'attractor' has no attribute {name!r}")
