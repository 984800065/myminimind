import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch import nn

from mini_deepseek.model.configuration_mini_deepseek import MiniDeepSeekConfig


class LinearCrossEntropyBase(nn.Module):
    """Common interface for LM-head + token CE implementations.

    All implementations consume flattened tensors:
    - `hidden_states`: (num_tokens, hidden_size)
    - `labels`: (num_tokens,)

    Keeping the interface unified lets the model code stay simple even though
    different backends have different calling conventions.
    """

    def forward(
        self,
        lm_head: nn.Linear,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class EagerLinearCrossEntropy(LinearCrossEntropyBase):
    """Reference implementation: materialize logits, then call PyTorch CE."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(
        self,
        lm_head: nn.Linear,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = lm_head(hidden_states)
        return self.ce(logits, labels)


class LigerLinearCrossEntropy(LinearCrossEntropyBase):
    """Liger fused implementation that avoids materializing full logits."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        lm_head: nn.Linear,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.ce(
            lm_head.weight,
            hidden_states,
            labels,
            bias=lm_head.bias,
        )


def build_linear_cross_entropy(
    config: MiniDeepSeekConfig,
    ignore_index: int = -100,
) -> LinearCrossEntropyBase:
    """Build the LM-head loss backend selected by config.

    Keep this intentionally small and explicit:
    - one config field decides the implementation
    - the model always calls one shared interface
    - adding a future Triton version is just one extra branch
    """

    implementation = config.linear_cross_entropy_implementation

    if implementation == "eager":
        return EagerLinearCrossEntropy(ignore_index=ignore_index)
    if implementation == "liger_fused":
        return LigerLinearCrossEntropy(ignore_index=ignore_index)

    raise ValueError(
        "Unsupported linear_cross_entropy_implementation="
        f"{implementation!r}. Expected one of: 'eager', 'liger_fused'."
    )
