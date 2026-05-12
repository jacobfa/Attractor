import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        with torch.autocast(enabled=False, device_type=x.device.type):
            return self._norm(x.float()).type_as(x) * self.weight

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)