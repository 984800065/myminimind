from typing import Any

import torch
from liger_kernel.transformers import LigerRMSNorm
from torch import nn

from myminimind.model.configuration_myminimind import MyMiniMindConfig


class RMSNorm(nn.Module):
    """Project-local eager RMSNorm implementation.

    This keeps the math explicit and easy to compare against other
    implementations such as Liger or a future Triton kernel.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps).type_as(x_fp32)
        return rms.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._rms(x) * self.weight.to(x.dtype)


class LayerNorm(nn.Module):
    """Simple eager LayerNorm over the last dimension."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = (x_fp32 - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x_fp32 - mean) / torch.sqrt(var + self.eps)
        x_hat = x_hat.to(x.dtype)
        return x_hat * self.weight.to(x.dtype) + self.bias.to(x.dtype)


def build_norm(
    config: MyMiniMindConfig,
    hidden_size: int | None = None,
    eps: float | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Build the norm module selected by `config.norm_implementation`.

    Keep this function intentionally simple:
    - one config field decides the implementation
    - `if / elif / else` makes the control flow obvious
    - adding a new norm later just means adding one more branch
    """

    hidden_size = config.hidden_size if hidden_size is None else hidden_size
    eps = config.rms_norm_eps if eps is None else eps
    implementation = config.norm_implementation

    if implementation == "layer_eager":
        return LayerNorm(hidden_size=hidden_size, eps=eps)
    if implementation == "rms_eager":
        return RMSNorm(hidden_size=hidden_size, eps=eps)
    if implementation == "rms_liger":
        return LigerRMSNorm(hidden_size=hidden_size, eps=eps, **kwargs)

    raise ValueError(
        "Unsupported norm_implementation="
        f"{implementation!r}. Expected one of: 'layer_eager', 'rms_eager', 'rms_liger'."
    )


__all__ = [
    "LayerNorm",
    "RMSNorm",
    "build_norm",
]
