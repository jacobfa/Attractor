import importlib
import pkgutil
from pathlib import Path

configs: list[dict] = []
name_to_config: dict[str, dict] = {}


def register_config(name: str, config: dict) -> dict:
    """Register a model configuration."""
    configs.append(config)
    name_to_config[name] = config
    return config


def _load_configs():
    """Auto-discover and load all configs from attractor/configs/"""
    configs_path = Path(__file__).parent.parent / "configs"
    if not configs_path.exists():
        return
    for subdir in configs_path.iterdir():
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for config_file in subdir.glob("*.py"):
            if config_file.name.startswith("_"):
                continue
            module_name = f"attractor.configs.{subdir.name}.{config_file.stem}"
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "configs"):
                    for conf in module.configs:
                        if "name" in conf:
                            register_config(conf["name"], conf)
            except Exception as e:
                print(f"Warning: Failed to load config from {module_name}: {e}")


_load_configs()

