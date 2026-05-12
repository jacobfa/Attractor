import torch
from torch import Tensor

from attractor.models.config import AnyConfig


class BaseMLP(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int = 0, in_features: int = 0) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = config.Linear(
            in_features, config.intermediate_size, bias=config.bias, init_method=config.init.fn("in_proj", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias, init_method=config.init.fn("out_proj", layer_id)
        )
        self.nonlin = config.Nonlin()
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.nonlin(self.fc(x)))


class GatedMLP(torch.nn.Module):
    def __init__(self, config: AnyConfig, layer_id: int, in_features: int = 0) -> None:
        super().__init__()
        self.config = config
        in_features = config.n_embd if in_features == 0 else in_features
        self.fc = config.Linear(
            in_features, config.intermediate_size * 2, bias=config.bias, init_method=config.init.fn("glu", layer_id)
        )
        self.proj = config.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias, init_method=config.init.fn("out_proj", layer_id)
        )
        self.nonlin = config.Nonlin()

    def forward(self, x: Tensor) -> Tensor:
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2
        return self.proj(x)