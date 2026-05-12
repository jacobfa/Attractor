import torch.nn as nn
import torch
from torch import Tensor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attractor.models.parcae.config import ParcaeConfig


class DiagonalInjection(nn.Module):
    """
    Parcae Diagonal Injection
    
    x_{t+1} = exp(- dt * A) * x_t + dt * B @ e
    """
    def __init__(
        self,
        config: "ParcaeConfig",
        layer_id: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        assert hasattr(config, "recurrent_embedding_dimension"), "Config missing 'recurrent_embedding_dimension'"
        assert getattr(config, "recurrent_embedding_dimension") is not None, "'recurrent_embedding_dimension' must not be None"
        state_dim = config.recurrent_embedding_dimension
        input_dim = config.n_embd

        self.A_log = nn.Parameter(torch.empty(state_dim))
        config.init.fn("ssm_A_log", layer_id)(self.A_log)
        self.A_log._no_weight_decay = True
        
        self.dt_bias = nn.Parameter(torch.empty(state_dim))
        config.init.fn("ssm_dt_bias", layer_id)(self.dt_bias)
        self.dt_bias._no_weight_decay = True

        # Always use identity initialization for B
        self.B = nn.Parameter(torch.empty(state_dim, input_dim))
        config.init.fn("ssm_B_identity", layer_id)(self.B)
        self.B._no_weight_decay = True

    def forward(self, x_t: Tensor, e: Tensor) -> Tensor:
        dt = nn.functional.softplus(self.dt_bias)
        A = torch.exp(self.A_log)
        decay = torch.exp(-dt * A)
        return x_t * decay + dt * (e @ self.B.T)

    @torch.no_grad()
    def get_spectral_norm(self) -> float:
        dt = nn.functional.softplus(self.dt_bias)
        A = torch.exp(self.A_log)
        decay = torch.exp(-dt * A)
        return decay.max().item()

    @torch.no_grad()
    def get_contraction_factor(self) -> float:
        dt = nn.functional.softplus(self.dt_bias)
        A = torch.exp(self.A_log)
        decay = torch.exp(-dt * A)
        return decay.mean().item()


class LinearInjection(nn.Module):
    def __init__(self, config: "ParcaeConfig"):
        super().__init__()
        self.config = config
        self.adapter = config.Linear(
            config.recurrent_embedding_dimension + config.n_embd,
            config.recurrent_embedding_dimension,
            bias=config.bias,
            init_method=config.init.fn("adapter", config.n_layers_in_prelude),
        )
    
    def forward(self, x: Tensor, input_embeds: Tensor) -> Tensor:
        return self.adapter(torch.cat([x, input_embeds], dim=-1))

class AdditiveInjection(nn.Module):
    def __init__(self, config: "ParcaeConfig"):
        super().__init__()
        assert config.recurrent_embedding_dimension == config.n_embd, "Recurrent embedding dimension and input dimension must be the same"
        self.config = config
    
    def forward(self, x: Tensor, input_embeds: Tensor) -> Tensor:
        return x + input_embeds

def _get_injection_method(config: "ParcaeConfig") -> nn.Module:
    assert hasattr(config, "injection_type"), "Config missing 'injection_type'"

    if config.injection_type == "diagonal":
        return DiagonalInjection(config)
    elif config.injection_type == "linear":
        return LinearInjection(config)
    elif config.injection_type == "add":
        return AdditiveInjection(config)
    else:
        raise ValueError(f"Invalid injection type: {config.injection_type}")