from dataclasses import dataclass
import torch
from attractor.models.config import Config
from attractor.models.gpt.init import GPTInit

@dataclass
class GPTConfig(Config):
    n_layer: int = 16
    tie_embeddings: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.init = GPTInit(
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

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from attractor.models.gpt import GPT
        return GPT(self, **kwargs)