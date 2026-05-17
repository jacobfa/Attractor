import torch
import math
from math import sqrt
from typing import Optional, Callable

from attractor.utils.init import Init, wrapped_trunc_normal


@torch.no_grad()
def init_adapter_identity(tensor, state_dim: int):
    d_out, d_in = tensor.shape
    assert d_in > state_dim, f"Input dim {d_in} must be > state_dim {state_dim}"
    tensor.zero_()
    identity_size = min(d_out, state_dim)
    for i in range(identity_size):
        tensor[i, i] = 1.0


@torch.no_grad()
def init_coda_identity(tensor):
    d_out, d_in = tensor.shape
    if d_out == d_in:
        tensor.zero_()
        for i in range(d_out):
            tensor[i, i] = 1.0
    else:
        torch.nn.init.orthogonal_(tensor)


@torch.no_grad()
def init_ssm_A_log(tensor):
    tensor.zero_()


@torch.no_grad()
def init_ssm_dt_bias(tensor, decay_target: float):
    target_product = -math.log(decay_target)
    dt_init = torch.full_like(tensor, target_product)
    inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))  # inverse softplus
    tensor.copy_(inv_dt)


@torch.no_grad()
def init_ssm_B_identity(tensor):
    d_out, d_in = tensor.shape
    if d_out == d_in:
        tensor.zero_()
        for i in range(d_out):
            tensor[i, i] = 1.0
    else:
        torch.nn.init.orthogonal_(tensor)


@torch.no_grad()
def init_ssm_B_scaled_orthogonal(tensor, ssm_decay: float):
    d_out, d_in = tensor.shape
    dt_init = -math.log(ssm_decay)
    scale = 1.0 / dt_init
    if d_out == d_in:
        tensor.zero_()
        for i in range(d_out):
            tensor[i, i] = scale
    else:
        torch.nn.init.orthogonal_(tensor)
        tensor.mul_(scale)


class ParcaeInit(Init):

    def _get_layer_init(self, name_of_layer: str, layer_idx: int, init_table: dict) -> Optional[Callable]:
        mu = self.mup_model_scaling_factor

        if "adapter_identity" in name_of_layer:
            parts = name_of_layer.split("_")
            state_dim = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else self.dim

            def init(tensor, state_dim=state_dim):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} as identity (state_dim={state_dim}).")
                init_adapter_identity(tensor, state_dim)
            return init

        elif "coda_identity" in name_of_layer:
            def init(tensor):
                if self.verbose:
                    d_out, d_in = tensor.shape
                    init_type = "identity" if d_out == d_in else "orthogonal"
                    print(f"Init layer {layer_idx} {name_of_layer} as {init_type} ({d_in} -> {d_out}).")
                init_coda_identity(tensor)
            return init

        elif "ssm_A_log" in name_of_layer:
            def init(tensor):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} as zeros (A=1).")
                init_ssm_A_log(tensor)
            return init

        elif "ssm_dt_bias" in name_of_layer:
            ssm_decay = init_table.get("ssm_decay", sqrt(1.0 / 5.0))

            def init(tensor, ssm_decay=ssm_decay):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} for decay={ssm_decay:.4f}.")
                init_ssm_dt_bias(tensor, ssm_decay)
            return init

        elif "ssm_B_scaled_orthogonal" in name_of_layer:
            ssm_decay = init_table.get("ssm_decay", sqrt(1.0 / 5.0))

            def init(tensor, ssm_decay=ssm_decay):
                if self.verbose:
                    d_out, d_in = tensor.shape
                    init_type = "identity" if d_out == d_in else "orthogonal"
                    dt_init = -math.log(ssm_decay)
                    print(f"Init layer {layer_idx} {name_of_layer} as {init_type} / dt_init={dt_init:.4f} ({d_in} -> {d_out}).")
                init_ssm_B_scaled_orthogonal(tensor, ssm_decay)
            return init

        elif "ssm_B_identity" in name_of_layer:
            def init(tensor):
                if self.verbose:
                    d_out, d_in = tensor.shape
                    init_type = "identity" if d_out == d_in else "orthogonal"
                    print(f"Init layer {layer_idx} {name_of_layer} as {init_type} ({d_in} -> {d_out}).")
                init_ssm_B_identity(tensor)
            return init

        elif "ssm_B" in name_of_layer:
            def init(tensor):
                fan_in = tensor.shape[1]
                std = sqrt(1.0 / (5.0 * fan_in))
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} with std={std:2.4f} (fan_in={fan_in}).")
                self.normal_(tensor, std=float(std))
            return init

        elif "adapter" in name_of_layer:
            std = sqrt(2 / (5 * self.dim)) / mu
            def init(tensor, std=std):
                if self.verbose:
                    print(f"Init layer {layer_idx} {name_of_layer} with std={std:2.4f} (trunc_normal).")
                wrapped_trunc_normal(tensor, std=float(std))
            return init

        return None
