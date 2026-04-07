import math
from typing import Protocol

import torch
import triton
from liger_kernel.ops.rope import _triton_rope
from torch import nn

from myminimind.model.configuration_myminimind import MyMiniMindConfig


class ApplyRotaryPosEmbFn(Protocol):
    """Callable signature shared by all q/k rotary application backends."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class RotaryEmbedding(nn.Module):
    freq: torch.Tensor

    def __init__(self, config: MyMiniMindConfig, dim: int):
        """
        显示的穿dim以适配各种不同的rope dim配置
        """
        super().__init__()
        self.config = config

        rope_scaling = getattr(self.config, "rope_scaling", None) or {}

        freq = self.compute_freq(rope_scaling, dim)
        self.attention_factor = rope_scaling.get("attention_factor", 1.0)

        self.register_buffer("freq", freq, persistent=False)

    def compute_freq(self, rope_scaling: dict, dim: int) -> torch.Tensor:
        rope_theta = self.config.rope_theta
        # freq.shape == (dim // 2, )
        assert dim % 2 == 0, "RoPE only supports even dimensions"
        freq = rope_theta ** (-torch.arange(0, dim, 2, dtype=torch.float) / dim)

        rope_type = rope_scaling.get("rope_type", "default")
        if rope_type == "default":
            freq = freq
        elif rope_type == "yarn":
            beta_fast = rope_scaling["beta_fast"]
            beta_slow = rope_scaling["beta_slow"]
            original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
            factor = rope_scaling["factor"]

            low = dim / 2 * math.log(original_max_position_embeddings / (2 * math.pi * beta_fast)) / math.log(rope_theta)
            high = dim / 2 * math.log(original_max_position_embeddings / (2 * math.pi * beta_slow)) / math.log(rope_theta)

            index = torch.arange(0, dim // 2, dtype=torch.float)
            ramp = torch.clamp((index - low) / max(high - low, 1e-5), 0, 1)
            freq = freq * ((1 - ramp) + ramp / factor)
        else:
            raise ValueError(f"Unsupported rope_scaling type: {rope_type}")
        
        return freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids.shape == (1, seq_len)
        # x.shape == (batch_size, num_heads, seq_len, dim)
        dim = x.shape[-1]
        assert dim % 2 == 0, "RoPE only supports even dimensions"
        # freq.shape == (dim // 2, )
        freq = self.freq

        # Force float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            # rotate_angle.shape == (1, seq_len, dim // 2)
            rotate_angle = position_ids[:, :, None] * freq[None, None, :]
            # double_rotate_angle.shape == (1, seq_len, dim)
            double_rotate_angle = torch.cat([rotate_angle, rotate_angle], dim=-1)
            # cos.shape == (1, seq_len, head_dim)
            cos = double_rotate_angle.cos() * self.attention_factor
            # sin.shape == (1, seq_len, head_dim)
            sin = double_rotate_angle.sin() * self.attention_factor

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LigerRopeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim)
        sin size: (1, seq_len, head_dim)
        """

        # transpose it back to the physical shape because Triton looks at the physical storage
        # note: q and k are incontiguous before the transformation and will become contiguous after transpose
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        batch_size, seq_len, n_q_head, head_dim = q.shape
        n_kv_head = k.shape[2]
        pad_hd = triton.next_power_of_2(head_dim)
        pad_n_q_head = triton.next_power_of_2(n_q_head)
        pad_n_kv_head = triton.next_power_of_2(n_kv_head)
        BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

        n_row = batch_size * seq_len

        # Added: enforce the tensor to be contiguous
        q = q.contiguous()
        k = k.contiguous()

        cos_batch_size = cos.shape[0]

        _triton_rope[(n_row,)](
            q,
            q.stride(1),
            k,
            k.stride(1),
            cos,
            cos.stride(-2),
            sin,
            sin.stride(-2),
            seq_len,
            batch_size,
            cos_batch_size,
            n_q_head,
            n_kv_head,
            head_dim,
            pad_n_q_head,
            pad_n_kv_head,
            pad_hd,
            BLOCK_SIZE=BLOCK_SIZE,
            BACKWARD_PASS=False,
        )

        ctx.save_for_backward(cos, sin)
        return q.transpose(1, 2), k.transpose(1, 2)


def liger_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def apply_rotary_pos_emb_interleave(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cos.shape == sin.shape == (1, seq_len, head_dim), 1 in dimension 0 is for broadcasting over batch_size

    # cos.shape == (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    # sin.shape == (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # q.shape == (batch_size, num_query_heads, seq_len, head_dim)
    # k.shape == (batch_size, num_key_value_heads, seq_len, head_dim)

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x.shape == (batch_size, num_heads, seq_len, head_dim)
        return torch.cat([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1)

    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def build_apply_rotary_pos_emb(config: MyMiniMindConfig) -> ApplyRotaryPosEmbFn:
    """Build the q/k rotary application function selected by `config.rope_implementation`.

    In this project RoPE has two parts:
    - `RotaryEmbedding`: prepares the frequency table and produces cos/sin
    - apply function: rotates q/k using either eager PyTorch code or a fused kernel
    """

    implementation = config.rope_implementation

    if implementation == "eager":
        return apply_rotary_pos_emb_interleave
    if implementation == "liger":
        return liger_rotary_pos_emb

    raise ValueError(
        "Unsupported rope_implementation="
        f"{implementation!r}. Expected one of: 'eager', 'liger'."
    )


__all__ = [
    "ApplyRotaryPosEmbFn",
    "RotaryEmbedding",
    "LigerRopeFunction",
    "apply_rotary_pos_emb_interleave",
    "liger_rotary_pos_emb",
    "build_apply_rotary_pos_emb",
]
