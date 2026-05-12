import torch
from attractor.utils.init import init_normal

class Linear(torch.nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, init_method=None
    ):
        self.init_method = init_method if init_method else init_normal(in_features)
        super().__init__(in_features, out_features, bias, device, dtype)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.init_method(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, **kwargs):
        return super().forward(input)


class Relu2(torch.nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.relu(x, inplace=self.inplace).pow(2).mul(0.5)  # mul just to be difficult? :<