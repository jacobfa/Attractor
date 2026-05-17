# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union, Callable, Self
from functools import partial
from collections import defaultdict

import torch
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

from attractor.utils import find_multiple
from attractor.models.registry import name_to_config, configs
from attractor.modules.basic import Linear, Relu2


@dataclass
class RoPESettings:
    use_rope: bool = True
    rope_condense_ratio: int = 1
    rope_base: int = 50_000


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)

    # Core
    block_size: int = 4096  # max_seq_len
    n_embd: int = 4096
    intermediate_size: Optional[int] = None  # type: ignore
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # type: ignore # new GQA notation oriented at Llama

    # Word+Pos Embedding
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    rope_settings: RoPESettings = field(default_factory=lambda: RoPESettings())
    use_abacus: bool = False
    # set randomize_positions_from to an integer greater than block_size to draw pos_ids from the entire range:
    randomize_positions_from: Optional[int] = None

    # Main blocks
    block_class_name: str = "TransformerPreNormBlock"
    norm_class_name: str = "LayerNorm"

    # Block details
    attn_impl: Literal["flash", "sdpa", "debug-skip"] = "flash"  # "flash" auto-selects FA3 on Hopper, SDPA elsewhere
    norm_eps: float = 1e-5
    mlp_class_name: str = "BaseMLP"
    nonlin_name: str = "GELU"  # draws from torch.nn first
    bias: bool = False
    qk_bias: bool = False
    init_strategy: str = "scaled"
    init_orthogonal: bool = False
    skip_initialization: bool = False

    mup_model_scaling_factor: int = 1  # use this to scale model width+lr+logit_scale

    use_fused_head: Literal["hhe", "cce", "full-triton", "pytorch"] = "pytorch"

    debias_attention: bool = False
    center_attention: bool = False

    clip_qkv: Optional[float] = None  # 8 in olmo1.7
    qk_norm: bool = False
    logit_softcap: Optional[float] = None  # e.g. 15.0 for nanochat-style logit bounding
    causal: bool = True  # set False for bidirectional / structured-output tasks (e.g. Sudoku)

    # Implementation handles
    activation_checkpoint_impl: str = "per-block"
    simple_ops: bool = False  # Choose naive implementations where flops can be traced more easily (never use in prod)
    strategy: str = "single"  # which device strategy is being used

    def __post_init__(self):
        # Convert dict to dataclass instances if needed
        if isinstance(self.rope_settings, dict):
            self.rope_settings = RoPESettings(**self.rope_settings)
        
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # Validate kv heads versus all heads
        if self.num_key_value_heads is not None:
            assert self.num_attention_heads % self.num_key_value_heads == 0
        else:
            self.num_key_value_heads: int = self.num_attention_heads
        assert self.n_embd % self.num_attention_heads == 0
        self.head_size = self.n_embd // self.num_attention_heads
        self.n_head = self.num_attention_heads  # for compatibility with config.py
        self.n_query_groups = self.num_key_value_heads  # for compatibility with config.py

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            self.intermediate_size: int = 4 * self.n_embd

        # SCALE architecture definition
        self.n_embd *= self.mup_model_scaling_factor
        self.intermediate_size *= self.mup_model_scaling_factor
        self.n_query_groups *= self.mup_model_scaling_factor
        self.num_key_value_heads *= self.mup_model_scaling_factor
        self.n_head *= self.mup_model_scaling_factor

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(config for config in configs if name == config["hf_config"]["name"])
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        
        if cls.__name__ == "Config":
            arch_class = conf_dict.get("architecture_class_name", "GPT")
            if arch_class == "GPT":
                from attractor.models.gpt.config import GPTConfig
                return GPTConfig.from_name(name, **kwargs)
            elif arch_class == "Parcae":
                from attractor.models.parcae.config import ParcaeConfig
                return ParcaeConfig.from_name(name, **kwargs)
            elif arch_class in ("Attractor", "EQLM"):
                from attractor.models.attractor.config import AttractorConfig
                return AttractorConfig.from_name(name, **kwargs)
        
        rope_settings = {}
        for key, value in kwargs.items():
            if key.startswith("rope_settings."):
                rope_key = key.split(".", 1)[1]
                rope_settings[rope_key] = value
            else:
                conf_dict[key] = value
        
        # Convert rope_settings from dict if present in conf_dict
        if "rope_settings" in conf_dict and isinstance(conf_dict["rope_settings"], dict):
            rope_dict = conf_dict.pop("rope_settings")
            rope_dict.update(rope_settings)  # Merge with any kwargs
            conf_dict["rope_settings"] = RoPESettings(**rope_dict)
        elif rope_settings:
            conf_dict["rope_settings"] = RoPESettings(**rope_settings)

        # Remove metadata fields that aren't config parameters
        conf_dict.pop("architecture_class_name", None)
        
        return cls(**conf_dict)

    def construct_model(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    @property
    def MLP(self) -> Type[torch.nn.Module]:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        from attractor.modules import mlp

        return getattr(mlp, self.mlp_class_name)

    @property
    def Linear(self) -> Type[torch.nn.Module]:
        if self.strategy == "axonn_tp" and not self.simple_ops:
            # Load different module for axonn tensor parallel
            from axonn.intra_layer import Linear as TensorParallelLinear

            return TensorParallelLinear
        else:
            return Linear

    @property
    def Block(self) -> Type[torch.nn.Module]:
        from attractor.modules import blocks

        return getattr(blocks, self.block_class_name)

    @property
    def Nonlin(self) -> Type[torch.nn.Module]:
        try:
            return getattr(torch.nn, self.nonlin_name)
        except AttributeError:
            if self.nonlin_name == "ReLU2":
                return Relu2
            else:
                raise ValueError(f"Could not identify nonlinearity {self.nonlin_name}")

    @property
    def Norm(self) -> Union[Type[torch.nn.Module], Callable]:
        if not self.simple_ops:
            try:
                from attractor.modules import norms

                norm_fn = getattr(norms, self.norm_class_name)
                if "Gemma" in self.name:
                    return partial(norm_fn, add_unit_offset=True)
                else:
                    return norm_fn
            except AttributeError:
                return getattr(torch.nn, self.norm_class_name)
        else:
            from attractor.modules import norms

            return norms.RMSNorm

    @property
    def checkpoint(self) -> Callable:
        """Run SAC at your own risk :<"""
        attn_ops = [
            torch.ops.aten._scaled_dot_product_efficient_attention.default,  # type: ignore
            torch.ops.aten._scaled_dot_product_flash_attention.default,  # type: ignore
        ]
        try:
            from flash_attn import flash_attn_func  # type: ignore

            attn_ops.append(flash_attn_func)
        except ImportError:
            pass
        ops_to_save = [
            torch.ops.aten.mm.default,  # type: ignore
            *attn_ops,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,  # type: ignore # from comms
        ]

        if "sac%" in self.activation_checkpoint_impl:
            frequency = int(self.activation_checkpoint_impl.split("sac%")[1][0])

            def _get_custom_policy(meta):
                def _custom_policy(ctx, func, *args, **kwargs):
                    mode = "recompute" if ctx.is_recompute else "forward"
                    mm_count_key = f"{mode}_mm_count"
                    if func == torch.ops.aten.mm.default:  # type: ignore
                        meta[mm_count_key] += 1
                    # Saves output of all compute ops, except every freq- mm
                    to_save = func in ops_to_save and not (
                        func == torch.ops.aten.mm.default and meta[mm_count_key] % frequency == 0  # type: ignore
                    )
                    return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

                return _custom_policy

            def context_fn():
                meta = defaultdict(int)
                return create_selective_checkpoint_contexts(_get_custom_policy(meta))

            return partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                context_fn=context_fn,
            )
        elif "sac-attn" in self.activation_checkpoint_impl:

            def policy_fn(ctx, op, *args, **kwargs):
                if op in attn_ops:
                    return CheckpointPolicy.MUST_SAVE
                else:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
            return partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                context_fn=context_fn,
            )

        elif "sac" in self.activation_checkpoint_impl:

            def policy_fn(ctx, op, *args, **kwargs):
                if op in ops_to_save:
                    return CheckpointPolicy.MUST_SAVE
                else:
                    return CheckpointPolicy.PREFER_RECOMPUTE

            context_fn = partial(create_selective_checkpoint_contexts, policy_fn)
            return partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=False,
                determinism_check="none",
                context_fn=context_fn,
            )
        elif "reentrant" in self.activation_checkpoint_impl:
            # Use reentrant checkpointing - better compatibility with closures and captured variables
            return partial(checkpoint, use_reentrant=True, preserve_rng_state=True)
        else:
            # returning context_fn can break inductor in funny ways, best not to provide it if not necessary
            return partial(checkpoint, use_reentrant=False, preserve_rng_state=False, determinism_check="none")

    # this is a bit of slop code, but ok for now
    def __getstate__(self):
        state = asdict(self)
        state["_class_name"] = self.__class__.__name__
        return state

    def __setstate__(self, state):
        if state["_class_name"] == self.__class__.__name__:
            rope_settings = RoPESettings(**state.pop("rope_settings"))
            state.pop("_class_name")
            self.__dict__.update(state)
            self.__dict__["rope_settings"] = rope_settings
            self.__post_init__()
        else:
            raise ValueError(f"Saved Architecture class name {state['_class_name']} does not match saved config.")


AnyConfig = Config