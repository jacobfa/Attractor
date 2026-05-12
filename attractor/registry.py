# type: ignore
from copy import deepcopy
import re
import importlib.util
from pathlib import Path

"""This config file contains only working / in-progress architecture definitions for model_dynamic.py
"""

# Default HuggingFace organization for model configs
DEFAULT_HF_ORG = "SandyResearch"

# Paths to search for config files
_MODEL_CONFIG_PATHS = [Path(__file__).parent / "configs"]

configs = []

###############
# Config file scanning
###############

def _natural_key(string_):
    """Natural sort key function for sorting config names."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    """Scan config directories for Python config files and load them."""
    global configs
    
    config_ext = ('.py',)
    config_files = []
    
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            # Skip the registry file itself
            if config_path.resolve() != Path(__file__).resolve():
                config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                # Recursively find all Python files
                config_files.extend(config_path.rglob(f'*{ext}'))
    
    # Remove duplicates and sort naturally
    config_files = sorted(set(config_files), key=lambda x: _natural_key(str(x)))
    
    loaded_configs = []
    for cf in config_files:
        try:
            # Skip __init__.py and __pycache__ files
            if cf.name.startswith('__') or '__pycache__' in str(cf):
                continue
            
            # Load the module
            spec = importlib.util.spec_from_file_location(f"config_{cf.stem}", cf)
            if spec is None or spec.loader is None:
                continue
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for config dict or list of configs
            # First check for a 'config' variable
            if hasattr(module, 'config'):
                cfg = module.config
                if isinstance(cfg, dict):
                    loaded_configs.append(cfg)
                elif isinstance(cfg, list):
                    loaded_configs.extend(cfg)
            # Check for 'configs' variable (list)
            elif hasattr(module, 'configs'):
                cfgs = module.configs
                if isinstance(cfgs, list):
                    loaded_configs.extend(cfgs)
            # Check for variables that look like config dicts (have 'name' key)
            else:
                for attr_name in dir(module):
                    if not attr_name.startswith('_'):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, dict) and 'name' in attr:
                            loaded_configs.append(attr)
                        elif isinstance(attr, list) and len(attr) > 0 and isinstance(attr[0], dict) and 'name' in attr[0]:
                            loaded_configs.extend(attr)
        except Exception as e:
            # Silently skip files that fail to load
            import warnings
            warnings.warn(f"Failed to load config from {cf}: {e}", UserWarning)
            continue
    
    # Add loaded configs to the main configs list
    if loaded_configs:
        configs.extend(loaded_configs)


def add_model_config_path(path):
    """Add a model config path or file and update registry.
    
    Args:
        path: Path to a directory or file to scan for configs
    """
    if not isinstance(path, Path):
        path = Path(path)
    
    if path not in _MODEL_CONFIG_PATHS:
        _MODEL_CONFIG_PATHS.append(path)
        _rescan_model_configs()


# Initial scan of config files
_rescan_model_configs()

# Build the name_to_config mapping
name_to_config = {config["name"]: config for config in configs}

