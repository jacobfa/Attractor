from dataclasses import dataclass
from typing import Literal, Optional

import torch

from attractor.models.config import Config
from attractor.models.eqlm.init import EQLMInit


@dataclass
class EQLMConfig(Config):
    # Backbone (before the fp head) — same role as GPT's whole stack.
    n_backbone_layers: int = 6
    n_fp_blocks: int = 1
    tie_embeddings: bool = True

    # Forward solver.
    solver: Literal["anderson", "fpi"] = "anderson"
    max_iter: int = 12
    min_iter: int = 0
    tol: float = 1e-3
    anderson_m: int = 5
    anderson_beta: float = 1.0

    # Backward (implicit-gradient) adjoint solver.
    backward_type: Literal["onestep", "anderson", "picard"] = "onestep"
    backward_max_iter: Optional[int] = None
    backward_min_iter: Optional[int] = None
    backward_tol: Optional[float] = None
    adjoint_grad_clip: Optional[float] = 1.0
    layer_scale_init: Optional[float] = 0.25
    gamma_max: Optional[float] = 1.0
    fp_lr_scale: float = 0.75
    fp_wd: float = 0.0

    recurrent_embedding_dimension: Optional[int] = None

    model_class_name: Literal["EQLM"] = "EQLM"

    def __post_init__(self):
        super().__post_init__()

        if self.backward_max_iter is None:
            self.backward_max_iter = self.max_iter
        if self.backward_min_iter is None:
            self.backward_min_iter = self.min_iter
        if self.backward_tol is None:
            self.backward_tol = self.tol

        if self.recurrent_embedding_dimension is None:
            self.recurrent_embedding_dimension = self.n_embd
        self.n_layers_in_prelude = int(self.n_backbone_layers)
        self.n_layers_in_recurrent_block = int(self.n_fp_blocks)
        self.n_layers_in_coda = 0

        self.mean_recurrence = int(self.max_iter)
        self.mean_backprop_depth = 1

        self.n_layer = (
            self.n_layers_in_prelude
            + self.n_layers_in_recurrent_block
            + self.n_layers_in_coda
        )

        self.init = EQLMInit(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            self.n_layer,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

    @property
    def recurrent_block_config(self) -> "EQLMConfig":
        return self

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from attractor.models.eqlm.eqlm import EQLM
        return EQLM(self, **kwargs)
