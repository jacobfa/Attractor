from dataclasses import dataclass
from typing import Literal, Optional

import torch

from attractor.models.config import Config
from attractor.models.attractor.init import AttractorInit


@dataclass
class AttractorConfig(Config):
    # Number of standard transformer blocks in the backbone (prelude).
    n_backbone_layers: int = 6
    # Number of weight-tied fixed-point blocks in the attractor head.
    n_fp_blocks: int = 1
    tie_embeddings: bool = True

    # Forward fixed-point solver settings.
    solver: Literal["anderson", "fpi"] = "anderson"
    max_iter: int = 12
    min_iter: int = 0
    tol: float = 1e-3
    anderson_m: int = 5
    anderson_beta: float = 1.0

    # Backward (implicit-gradient) adjoint settings.
    backward_type: Literal["onestep", "anderson", "picard"] = "onestep"
    backward_max_iter: Optional[int] = None
    backward_min_iter: Optional[int] = None
    backward_tol: Optional[float] = None
    # Clips J^T v to this multiple of ||v|| in the one-step adjoint to keep
    # the Neumann-1 approximation safe when the head drifts less contractive.
    adjoint_grad_clip: Optional[float] = 1.0

    # LayerScale gate init: effective gamma at init = layer_scale_init.
    layer_scale_init: Optional[float] = 0.25
    gamma_max: Optional[float] = 1.0

    # LR and weight-decay overrides for the attractor head parameters.
    fp_lr_scale: float = 0.75
    fp_wd: float = 0.0

    recurrent_embedding_dimension: Optional[int] = None

    model_class_name: Literal["Attractor"] = "Attractor"

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

        # Aliases required by parcae's training infra.
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

        self.init = AttractorInit(
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
    def recurrent_block_config(self) -> "AttractorConfig":
        return self

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from attractor.models.attractor.attractor import Attractor
        return Attractor(self, **kwargs)
