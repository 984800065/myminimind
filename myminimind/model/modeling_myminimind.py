import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.moe import use_experts_implementation
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, create_causal_mask, eager_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from .configuration_myminimind import MyMiniMindConfig

SUPPORTED_EXPERTS_IMPLEMENTATIONS = {"eager", "grouped_mm", "batched_mm"}


def apply_rotary_pos_emb_interleave(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: int | None = None, unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    # cos.shape == (batch_size, 1, seq_len, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    # sin.shape == (batch_size, 1, seq_len, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # q.shape == (batch_size, num_query_heads, seq_len, head_dim)
    # k.shape == (batch_size, num_key_value_heads, seq_len, head_dim)

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x.shape == (batch_size, num_heads, seq_len, head_dim)
        return torch.cat([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1)

    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def myminimind_gqa_eager_attention_forward(
    self: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    
    # q.shape == (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, query_len, head_dim = q.shape

    # k.shape == v.shape == (batch_size, num_heads, seq_len, head_dim)
    group_num = k.shape[1]
    group_size = num_heads // group_num

    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)

    if scaling is None:
        scaling = head_dim ** -0.5
        assert scaling

    # attn_score.shape == (batch_size, num_heads, seq_len, seq_len)
    attn_score: torch.Tensor = torch.einsum("bhid,bhjd->bhij", q, k) * scaling

    if attention_mask is None:
        attn_score = attn_score
    else:
        if attention_mask.dtype == torch.bool:
            # attention_mask.shape == (batch_size, seq_len)
            attn_score = attn_score.masked_fill(attention_mask[:, None, None, :].logical_not(), float("-inf"))
        else:
            attn_score = attn_score + attention_mask

    attn_score = attn_score.softmax(dim=-1)
    attn_score = F.dropout(attn_score, p=dropout, training=self.training)

    # output.shape == (batch_size, num_heads, seq_len, head_dim)
    output = torch.einsum("bhij,bhjd->bhid", attn_score, v)
    return output, attn_score


ALL_ATTENTION_FUNCTIONS.register("my_gqa", myminimind_gqa_eager_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("my_gqa", eager_mask)


def myminimind_mla_attention_forward(*args, **kwargs):
    raise NotImplementedError("MyMLA handles attention internally and should not dispatch through ALL_ATTENTION_FUNCTIONS.")


ALL_ATTENTION_FUNCTIONS.register("my_mla", myminimind_mla_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("my_mla", eager_mask)


class MyBaseModelOutputWithPast(BaseModelOutputWithPast):
    aux_loss: torch.Tensor | None = None

    def __init__(
        self,
        aux_loss: torch.Tensor | None = None,
        mtp_hidden_states: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aux_loss = aux_loss
        self.mtp_hidden_states = mtp_hidden_states


class MyCausalLMOutputWithPast(CausalLMOutputWithPast):
    aux_loss: torch.Tensor | None = None

    def __init__(
        self,
        aux_loss: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aux_loss = aux_loss


class RMSNorm(nn.Module):
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


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MyMiniMindConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_seq_len
        self.original_max_seq_len = config.max_seq_len

        self.config = config

        rope_scaling = getattr(self.config, "rope_scaling", None) or {}
        self.rope_type = rope_scaling.get("rope_type", "default")
        rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: MyMiniMindConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        base = config.rope_theta

        # adjust for multihead attention
        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        dim = (
            config.mla_qk_rope_head_dim
            if getattr(config, "attention_type", "gqa") == "mla"
            else config.hidden_size // config.num_attention_heads
        )

        attention_factor = float(rope_scaling.get("attention_factor", 1.0))

        # inv_freq.shape == (dim // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    comput_default_rope_parameters = compute_default_rope_parameters

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # inv_freq_expanded.shape == (batch_size, dim // 2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)

        # position_ids.shape == (batch_size, seq_len)
        # position_ids_expanded.shape ==(batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        # Force float32
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs.shape == (batch_size, dim // 2, seq_len)
            freqs = torch.einsum("bdi,bis->bds", inv_freq_expanded, position_ids_expanded)
            # freqs.shape == (batch_size, seq_len, dim // 2)
            freqs = freqs.transpose(1, 2)
            emb = torch.cat([freqs, freqs], dim=-1)
            # cos.shape == (batch_size, seq_len, head_dim)
            cos = emb.cos() * self.attention_scaling
            # sin.shape == (batch_size, seq_len, head_dim)
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GroupQueryAttention(nn.Module):
    def __init__(
        self,
        config: MyMiniMindConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // self.num_heads

        self.group_num = config.group_num
        assert self.num_heads % self.group_num == 0, "num_heads must be divisible by group_num"
        self.num_key_value_heads = self.group_num
        self.group_size = self.num_heads // self.group_num

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.attention_dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)
        self.scaling = self.head_dim**-0.5

        self.is_flash_attention = config.flash_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor | None:
        # hidden_states.shape == (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = hidden_states.shape
        num_heads = self.num_heads
        num_key_value_heads = self.num_key_value_heads
        head_dim = self.head_dim

        # (batch_size, seq_len, num_heads * head_dim)
        q: torch.Tensor = self.q_proj(hidden_states)
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        # (batch_size, seq_len, num_key_value_heads * head_dim)
        k: torch.Tensor = self.k_proj(hidden_states)
        # (batch_size, num_key_value_heads, seq_len, head_dim)
        k = k.reshape(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

        # (batch_size, seq_len, num_key_value_heads * head_dim)
        v: torch.Tensor = self.v_proj(hidden_states)
        # (batch_size, num_key_value_heads, seq_len, head_dim)
        v = v.reshape(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

        # cos.shape == sin.shape == (batch_size, seq_len, head_dim)
        cos, sin = position_embeddings
        # q.shape == (batch_size, num_heads, seq_len, head_dim)
        # k.shape == (batch_size, num_key_value_heads, seq_len, head_dim)
        q, k = apply_rotary_pos_emb_interleave(q, k, cos=cos, sin=sin, position_ids=None, unsqueeze_dim=1)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # attention_interface = get_attention_interface(getattr(self.config, "_attn_implementation", "eager"))
        # output, _ = attention_interface(
        #     self,
        #     q,
        #     k,
        #     v,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     **kwargs,
        # )
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        output: torch.Tensor
        # output.shape == (batch_size, num_heads, seq_len, head_dim)
        output, attn_weight = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        # (batch_size, seq_len, num_heads * head_dim) == (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # (batch_size, seq_len, hidden_size)
        output = self.residual_dropout(self.out_proj(output))
        return output


class MyMLA(nn.Module):
    def __init__(self, config: MyMiniMindConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // 1

        self.q_lora_rank = config.mla_q_lora_rank
        self.kv_lora_rank = config.mla_kv_lora_rank

        self.qk_nope_head_dim = config.mla_qk_nope_head_dim
        self.qk_rope_head_dim = config.mla_qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.mla_v_head_dim

        if self.q_lora_rank > 0:
            self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
            self.q_proj = None
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.qk_head_dim, bias=False)
            self.wq_a = None
            self.q_norm = None
            self.wq_b = None

        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.attention_dropout = config.dropout
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.scale_fmt = config.scale_fmt

        self.dequant_wkv_b = None

    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_proj is not None:
            return self.q_proj(hidden_states)

        assert self.wq_a is not None
        assert self.q_norm is not None
        assert self.wq_b is not None
        return self.wq_b(self.q_norm(self.wq_a(hidden_states)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # hidden_states.shape == (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = hidden_states.shape
        n_heads = self.n_heads
        qk_head_dim = self.qk_head_dim

        # q.shape == (batch_size, seq_len, num_heads * qk_head_dim)
        q: torch.Tensor = self._project_query(hidden_states)
        # q.shape == (batch_size, num_heads, seq_len, qk_head_dim)
        q = q.view(batch_size, seq_len, n_heads, qk_head_dim).transpose(1, 2)
        # q_nope.shape == (batch_size, num_heads, seq_len, qk_nope_head_dim)
        # q_pe.shape == (batch_size, num_heads, seq_len, qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # kv.shape == (batch_size, seq_len, kv_lora_rank + qk_rope_head_dim)
        kv = self.wkv_a(hidden_states)
        # kv.shape == (batch_size, seq_len, kv_lora_rank)
        # k_pe.shape == (batch_size, seq_len, qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        
        cos, sin = position_embeddings
        # q_pe.shape == (batch_size, num_heads, seq_len, qk_rope_head_dim)
        # k_pe.shape == (batch_size, 1, seq_len, qk_rope_head_dim)
        q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe.unsqueeze(1), cos, sin, position_ids=None, unsqueeze_dim=1)
        # k_pe.shape == (batch_size, seq_len, qk_rope_head_dim)
        k_pe = k_pe.squeeze(1)

        total_seq_len = seq_len
        if past_key_values is not None:
            # kv.shape == (batch_size, total_seq_len, kv_lora_rank)
            # k_pe.shape == (batch_size, total_seq_len, qk_rope_head_dim)
            kv, k_pe = past_key_values.update(kv, k_pe, self.layer_idx)
            total_seq_len = kv.shape[1]

        # prefill
        if seq_len > 1:
            # q.shape == (batch_size, num_heads, seq_len, qk_head_dim)
            q = torch.cat([q_nope, q_pe], dim=-1)
            # kv.shape == (batch_size, total_seq_len, n_heads * (qk_nope_head_dim + v_head_dim))
            kv: torch.Tensor = self.wkv_b(kv)
            # kv.shape == (batch_size, n_heads, total_seq_len, qk_nope_head_dim + v_head_dim)
            kv = kv.view(batch_size, total_seq_len, n_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
            # k_nope.shape == (batch_size, num_heads, total_seq_len, qk_nope_head_dim)
            # v.shape == (batch_size, num_heads, total_seq_len, v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # k.shape == (batch_size, num_heads, total_seq_len, qk_nope_head_dim + qk_rope_head_dim) == (batch_size, num_heads, total_seq_len, qk_head_dim)
            k = torch.cat([k_nope, k_pe[:, None, :, :].expand(-1, n_heads, -1, -1)], dim=-1)

            # scores.shape == (batch_size, num_heads, seq_len, total_seq_len)
            scores: torch.Tensor = torch.einsum("bhid,bhjd->bhij", q, k) * self.softmax_scale
            # scores.shape == (batch_size, num_heads, seq_len, total_seq_len)
            scores = scores + attention_mask if attention_mask is not None else scores
            scores = scores.softmax(-1)
            scores = F.dropout(scores, p=self.attention_dropout, training=self.training)

            # x.shape == (batch_size, num_heads, seq_len, v_head_dim)
            x = torch.einsum("bhij,bhjd->bhid", scores, v)
        else:
            # wkv_b.shape == (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
            wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            # wkv_b.shape == (num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
            # q_nope.shape == (batch_size, num_heads, seq_len, kv_lora_rank)
            q_nope = torch.einsum("bhid,hdc->bhic", q_nope, wkv_b[:, :self.qk_nope_head_dim])

            # scores.shape == (batch_size, num_heads, seq_len, total_seq_len)
            scores = (torch.einsum("bhic,bjc->bhij", q_nope, kv[:batch_size]) + 
                      torch.einsum("bhir,bjr->bhij", q_pe, k_pe[:batch_size])) * self.softmax_scale
            
            # scores.shape == (batch_size, num_heads, seq_len, total_seq_len)
            scores = scores + attention_mask if attention_mask is not None else scores
            # scores.shape == (batch_size, num_heads, seq_len, total_seq_len)
            scores = scores.softmax(-1)
            scores = F.dropout(scores, p=self.attention_dropout, training=self.training)
            # x.shape == (batch_size, num_heads, seq_len, kv_lora_rank)
            x = torch.einsum("bhij,bjc->bhic", scores, kv[:batch_size])
            # x.shape == (batch_size, num_heads, seq_len, v_head_dim)
            x = torch.einsum("bhic,hdc->bhid", x, wkv_b[:, -self.v_head_dim:])

        # x.shape == (batch_size, seq_len, num_heads * v_head_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # x.shape == (batch_size, seq_len, hidden_size)
        x = self.wo(x)
        return x


def build_attention_layer(config: MyMiniMindConfig, layer_idx: int) -> nn.Module:
    attention_type = getattr(config, "attention_type", "gqa")
    if attention_type == "gqa":
        return GroupQueryAttention(config, layer_idx)
    if attention_type == "mla":
        # return MultiHeadLatentAttention(config, layer_idx)
        return MyMLA(config, layer_idx)
    raise ValueError(f"Unsupported attention_type={attention_type!r}.")


class GLU_FFN(nn.Module):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class FeedForward(nn.Module):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__()
        self.glu_ffn = GLU_FFN(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # multi return to compatible with MoEFeedForward
        return self.glu_ffn(x), x.new_tensor(0.0)


class MoEGate(nn.Module):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_routed_experts = config.num_routed_experts

        self.scoring_function = config.scoring_function
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob

        # (num_routed_experts, hidden_size)
        self.weight = nn.Parameter(torch.empty((self.num_routed_experts, config.hidden_size)))
        self.reset_parameter()

    def reset_parameter(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        # hidden_states.shape == (batch_size, seq_len, hidden_size)

        # (batch_size, seq_len, num_routed_experts)
        logits = F.linear(hidden_states, self.weight, None).float()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        if self.scoring_function == "softmax":
            # (batch_size, seq_len, num_routed_experts)
            scores = logits.softmax(dim=-1).to(hidden_states.dtype)
        else:
            raise NotImplementedError(f"unsupportable scoring function for MoE gating: {self.scoring_function}")

        # (batch_size, seq_len, top_k), (batch_size, seq_len, top_k)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                # DeepSeek-v2 style auxiliary loss
                # (batch_size, num_routed_experts)
                p = scores.mean(dim=1)

                # (batch_size, seq_len, num_routed_experts)
                f = torch.zeros((batch_size, seq_len, self.num_routed_experts), device=hidden_states.device)
                src = torch.ones(topk_idx.shape, device=hidden_states.device, dtype=f.dtype)
                f.scatter_add_(dim=2, index=topk_idx, src=src)
                # (batch_size, seq_len, num_routed_experts)
                f = self.num_routed_experts / (self.top_k * seq_len) * f
                # (batch_size, num_routed_experts)
                f = f.mean(dim=1)

                # (batch_size, )
                aux_loss = self.alpha * (f * p).sum(dim=1)
                # ()
                aux_loss = aux_loss.mean()
            else:
                # Switch Transformer style auxiliary loss
                # (batch_size * seq_len, num_routed_experts)
                p = scores.reshape(-1, self.num_routed_experts)
                # (num_routed_experts, )
                p = p.mean(dim=0)

                # (batch_size, seq_len, num_routed_experts)
                f = torch.zeros((batch_size, seq_len, self.num_routed_experts), device=hidden_states.device)
                src = torch.ones(topk_idx.shape, device=hidden_states.device, dtype=f.dtype)
                f.scatter_add_(dim=2, index=topk_idx, src=src)
                # (batch_size * seq_len, num_routed_experts)
                f = f.reshape(-1, self.num_routed_experts)
                # (num_routed_experts, )
                f = f.mean(dim=0)

                # ()
                aux_loss = self.alpha * self.num_routed_experts * (f * p).sum()
        else:
            # ()
            aux_loss = scores.new_tensor(0.0)

        return topk_idx, topk_weight, aux_loss


@use_experts_implementation
class MyMiniMindExperts(nn.ModuleList):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__([GLU_FFN(config) for _ in range(config.num_routed_experts)])
        self.num_experts = config.num_routed_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            hit_experts = {int(idx.item()) for idx in expert_hit.flatten()}

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0].item())
            if expert_idx == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            routing_weights = top_k_weights[token_idx, top_k_pos]
            keep_mask = routing_weights > 0
            if not keep_mask.any():
                continue

            token_idx = token_idx[keep_mask]
            routing_weights = routing_weights[keep_mask]
            current_state = hidden_states[token_idx]
            current_hidden_states = self[expert_idx](current_state)
            current_hidden_states = current_hidden_states * routing_weights[:, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        if self.training:
            for expert_idx, expert in enumerate(self):
                if expert_idx not in hit_experts:
                    final_hidden_states = final_hidden_states + sum(p.sum() for p in expert.parameters()) * 0.0

        return final_hidden_states


class MoEFeedForward(nn.Module):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = MyMiniMindExperts(config)
        self.gate = MoEGate(config)
        self.shared_experts = nn.ModuleList([GLU_FFN(config) for _ in range(config.num_shared_experts)])
        self.capacity_factor = config.capacity_factor

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        top_k = self.config.num_experts_per_token

        # topk_idx.shape == (batch_size, seq_len, top_k)
        # topk_weights.shape == (batch_size, seq_len, top_k)
        # aux_loss.shape == ()
        topk_idx: torch.Tensor
        topk_weight: torch.Tensor
        aux_loss: torch.Tensor
        topk_idx, topk_weight, aux_loss = self.gate(x)

        # avoid bugs
        topk_idx = topk_idx.contiguous()
        topk_weight = topk_weight.contiguous()

        expert_capacity = math.ceil(batch_size * seq_len * top_k / self.config.num_routed_experts * self.capacity_factor)

        # (batch_size * seq_len, hidden_size)
        flat_x = x.reshape(-1, hidden_size)

        # (batch_size * seq_len * top_k, )
        flat_expert_indices = topk_idx.reshape(-1)

        if self.config.norm_topk_prob:
            # (batch_size, seq_len, top_k)
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            topk_weight = topk_weight.to(flat_x.dtype)
        # (batch_size * seq_len * top_k, )
        flat_expert_weights = topk_weight.reshape(-1)

        # only difference between new_token and token is that new_token // top_k == token
        # (batch_size * seq_len * top_k, )
        new_token_indices_sorted_by_expert_id = torch.argsort(flat_expert_indices)

        # (num_routed_experts, )
        num_token_per_expert = flat_expert_indices.bincount(minlength=self.config.num_routed_experts).cumsum(dim=0)

        keep_assignment = torch.zeros_like(flat_expert_weights, dtype=torch.bool)
        start_index = 0
        for expert_index in range(self.config.num_routed_experts):
            next_start_index = num_token_per_expert[expert_index].item()
            selected_assignment = new_token_indices_sorted_by_expert_id[start_index : min(start_index + expert_capacity, next_start_index)]
            start_index = next_start_index
            keep_assignment[selected_assignment] = True

        flat_expert_weights = flat_expert_weights * keep_assignment.to(flat_expert_weights.dtype)
        topk_weight = flat_expert_weights.reshape(-1, top_k)
        flat_y = self.experts(flat_x, topk_idx.reshape(-1, top_k), topk_weight)

        # shared expert parallel simulation
        if len(self.shared_experts) > 0:
            scale = 1.0 / len(self.shared_experts)
            for shared_expert in self.shared_experts:
                flat_y += shared_expert(flat_x) * scale

        # (batch_size * seq_len, hidden_size)
        y = flat_y.reshape(batch_size, seq_len, hidden_size)

        self.aux_loss = aux_loss
        # original return function
        # return y

        # my return function
        return y, self.aux_loss


class MyMiniMindDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MyMiniMindConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attention = build_attention_layer(config, layer_idx)

        self.layer_idx = layer_idx
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: Cache | None = None, use_cache: bool | None = False, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attention(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss


class MyMiniMindMTPModule(nn.Module):
    def __init__(self, config: MyMiniMindConfig, mtp_module_idx: int, max_main_model_layer_idx: int):
        super().__init__()
        self.config = config
        # mtp_module_idx >= 1
        self.mtp_module_idx = mtp_module_idx
        assert self.mtp_module_idx >= 1, "mtp_module_idx should start from 1"
        self.layer_idx = max_main_model_layer_idx + mtp_module_idx

        self.embed_dropout = nn.Dropout(config.dropout)
        self.mtp_layernorm = RMSNorm(config.hidden_size * 2, eps=config.rms_norm_eps)
        self.linear_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.decoder_layer = MyMiniMindDecoderLayer(config, layer_idx=self.layer_idx)

        self.rotary_emb = RotaryEmbedding(config=config)
    
    def forward(
        self, 
        prev_embeds: torch.Tensor,
        cur_embeds: torch.Tensor,
        original_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        original_attention_mask: torch.Tensor,
        original_position_ids: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.training, "MTP module is only for training and should not be used during inference"
        # prev_embeds.shape == (batch_size, seq_len, hidden_size)
        # cur_embeds.shape == (batch_size, seq_len, hidden_size)

        # input_embeds.shape == (batch_size, seq_len - mtp_module_idx, hidden_size * 2)
        input_embeds = torch.cat([prev_embeds[:, self.mtp_module_idx - 1:-1], cur_embeds[:, self.mtp_module_idx:]], dim=-1)

        # hidden_states.shape == (batch_size, seq_len - mtp_module_idx, hidden_size * 2)
        hidden_states = self.embed_dropout(input_embeds)

        # hidden_states.shape == (batch_size, seq_len - mtp_module_idx, hidden_size)
        hidden_states = self.linear_proj(self.mtp_layernorm(hidden_states))

        # original_attention_mask.shape == (batch_size, 1, query_length, key_length)
        # original_position_embeddings[0].shape == original_position_embeddings[1].shape == (batch_size, seq_len, head_dim)
        # original_position_ids.shape == (1, seq_len)

        # output_hidden_states.shape == (batch_size, seq_len, hidden_size)
        output_hidden_states = torch.zeros_like(prev_embeds)
        hidden_states, layer_aux_loss = self.decoder_layer(
            hidden_states,
            attention_mask=original_attention_mask[:, :, self.mtp_module_idx:, self.mtp_module_idx:],
            position_embeddings=(original_position_embeddings[0][:, self.mtp_module_idx:, :], original_position_embeddings[1][:, self.mtp_module_idx:, :]),
            position_ids=original_position_ids[:, self.mtp_module_idx:],
            past_key_values=None,
            use_cache=False,
            **kwargs,
        )
        output_hidden_states[:, self.mtp_module_idx:, :] = hidden_states

        return output_hidden_states, layer_aux_loss


class MyMinimindPreTrainedModel(PreTrainedModel):
    config: MyMiniMindConfig
    config_class = MyMiniMindConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["MyMiniMindDecoderLayer"]

    def get_correct_experts_implementation(self, requested_experts: str | None) -> str:
        requested_experts = "eager" if requested_experts is None else requested_experts
        if requested_experts not in SUPPORTED_EXPERTS_IMPLEMENTATIONS:
            message = f'Specified `experts_implementation="{requested_experts}"` is not supported. The only possible arguments are `experts_implementation="eager"`, `"experts_implementation=grouped_mm"` and `"experts_implementation=batched_mm"`.'
            raise ValueError(message)

        parent_getter = getattr(PreTrainedModel, "get_correct_experts_implementation", None)
        if parent_getter is None:
            if requested_experts != "eager":
                raise ValueError('This Transformers version does not provide MoE dispatch helpers. Only `experts_implementation="eager"` is supported.')
            return requested_experts

        return parent_getter(self, requested_experts)


class MyMiniMindModel(MyMinimindPreTrainedModel):
    _supports_attention_backend = True

    def __init__(self, config: MyMiniMindConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([MyMiniMindDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        self.mtp_level = config.mtp_level
        if self.mtp_level > 0 and not self.training:
            raise ValueError("MTP modules are only for training and should not be used during inference")
        self.mtp_layers = nn.ModuleList([MyMiniMindMTPModule(config, mtp_module_idx=i, max_main_model_layer_idx=self.num_hidden_layers - 1) for i in range(1, config.mtp_level + 1)])
        
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        input_embeds: torch.FloatTensor | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ) -> MyBaseModelOutputWithPast | tuple[torch.Tensor, Cache | None, torch.Tensor]:
        # input_ids.shape == (batch_size, seq_len)
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_embeds is None:
            # input_embeds.shape == (batch_size, seq_len, hidden_dim)
            input_embeds = self.embed_tokens(input_ids)
            assert input_embeds is not None

        batch_size, seq_len = input_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            # cache_position.shape == (seq_len, )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len,
                device=input_embeds.device,
                dtype=torch.long,
            )

        if position_ids is None:
            # position_ids.shape == (1, seq_len)
            position_ids = cache_position.unsqueeze(0)
            # 如果你后面的 rotary 实现明确要求 batch 维完全展开，也可以用：
            # position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # causal_mask.shape == (batch_size, 1, query_length, key_length)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        assert causal_mask is not None

        # hidden_states.shape == (batch_size, seq_len, hidden_dim)
        hidden_states: torch.Tensor = self.dropout(input_embeds)

        # [cos, sin]
        # cos.shape == sin.shape == (batch_size, seq_len, head_dim)
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = self.rotary_emb(hidden_states, position_ids=position_ids)
        aux_loss = torch.tensor(0.0, device=hidden_states.device)

        mtp_hidden_states = hidden_states.new_zeros((self.mtp_level + 1, *hidden_states.shape))

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            # hidden_states.shape == (batch_size, seq_len, hidden_size)
            hidden_states, layer_aux_loss = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            aux_loss += layer_aux_loss
        
        mtp_hidden_states[0] = self.norm(hidden_states)

        for i, mtp_layer in enumerate(self.mtp_layers[:self.mtp_level]):
            # mtp_layer_idx \in [1, mtp_level]
            mtp_layer_idx = i + 1

            # input_embeds.shape == (batch_size, seq_len, hidden_size)
            # attention_mask.shape == (batch_size, 1, query_length, key_length)
            # position_embeddings[0].shape == position_embeddings[1].shape == (batch_size, seq_len, head_dim)
            # position_ids.shape == (1, seq_len)
            # hidden_states.shape == (batch_size, seq_len, hidden_size)
            hidden_states, layer_aux_loss = mtp_layer(
                prev_embeds=hidden_states,
                cur_embeds=input_embeds,
                original_position_embeddings=position_embeddings,
                original_attention_mask=causal_mask,
                original_position_ids=position_ids,
                **kwargs,
            )
            aux_loss += layer_aux_loss

            mtp_hidden_states[mtp_layer_idx] = self.norm(hidden_states)

        if not return_dict:
            return mtp_hidden_states, past_key_values, aux_loss

        return MyBaseModelOutputWithPast(
            last_hidden_state=mtp_hidden_states[0],
            past_key_values=past_key_values,
            aux_loss=aux_loss,
            mtp_hidden_states=mtp_hidden_states,
        )


class MyMiniMindForCausalLM(MyMinimindPreTrainedModel, GenerationMixin):
    config_class = MyMiniMindConfig
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MyMiniMindConfig):
        super().__init__(config)
        self.model = MyMiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.mtp_level = config.mtp_level
        self.mtp_lambda = config.mtp_lambda

        if self.mtp_level > 0 and not self.training:
            raise ValueError("MTP modules are only for training and should not be used during inference")
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings
        self.tie_weights()

    def tie_weights(
        self,
        missing_keys: set[str] | None = None,
        recompute_mapping: bool = True,
    ) -> None:
        if not getattr(self.config, "tie_word_embeddings", False):
            return
        if recompute_mapping or not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = dict(self._tied_weights_keys)
        self.lm_head.weight = self.model.embed_tokens.weight
        if missing_keys is not None:
            missing_keys.discard("lm_head.weight")

    def forward(
        self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None, use_cache: bool = False, logits_to_keep: int | torch.Tensor = 0, return_dict: bool | None = None, **kwargs
    ) -> MyCausalLMOutputWithPast | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_outputs: tuple | MyBaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs,
        )
        if return_dict:
            assert isinstance(model_outputs, MyBaseModelOutputWithPast)
            hidden_states = model_outputs.mtp_hidden_states
            aux_loss = model_outputs.aux_loss
            present_key_values = model_outputs.past_key_values
        else:
            assert isinstance(model_outputs, tuple)
            hidden_states, present_key_values, aux_loss = model_outputs
        # hidden_states.shape == (mtp_level + 1, batch_size, seq_len, hidden_size)
        assert hidden_states is not None

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = torch.scalar_tensor(0.0, device=hidden_states.device)
            for mtp_step in range(self.mtp_level + 1):
                # logits.shape == (batch_size, seq_len - mtp_step, vocab_size)
                logits: torch.Tensor = self.lm_head(hidden_states[mtp_step][:, mtp_step:, :])
                # shift_logits.shape == (batch_size, seq_len - mtp_step - 1, vocab_size)
                shift_logits = logits[..., :-1, :].contiguous()

                # labels.shape == (batch_size, seq_len)
                # shift_labels.shape == (batch_size, seq_len - mtp_step - 1)
                shift_labels = labels[:, mtp_step + 1:].contiguous()
                # label中句子完成之后的padding token的id被赋值成了-100，因此这些token不计入损失
                loss = loss + F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            
            loss = self.mtp_lambda * loss / (self.mtp_level + 1)

            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"loss is not finite, shift_logits: {shift_logits}, shift_labels: {shift_labels}")

        # logits.shape == (batch_size, seq_len, vocab_size)
        logits: torch.Tensor = self.lm_head(hidden_states[0])
        if not return_dict:
            output = (logits, present_key_values, hidden_states, aux_loss)
            return ((loss,) + output) if loss is not None else output

        output = MyCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=present_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss

        return output


def register_myminimind_for_auto_class() -> None:
    """Enable save_pretrained to emit auto_map metadata for custom-code loading."""
    if getattr(MyMiniMindConfig, "_auto_class", None) is None:
        MyMiniMindConfig.register_for_auto_class()
    if getattr(MyMiniMindModel, "_auto_class", None) is None:
        MyMiniMindModel.register_for_auto_class("AutoModel")
    if getattr(MyMiniMindForCausalLM, "_auto_class", None) is None:
        MyMiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")


__all__ = [
    "FeedForward",
    "GLU_FFN",
    "GroupQueryAttention",
    "MoEFeedForward",
    "MoEGate",
    "MyBaseModelOutputWithPast",
    "MyCausalLMOutputWithPast",
    "MyMiniMindDecoderLayer",
    "MyMiniMindExperts",
    "MyMiniMindForCausalLM",
    "MyMiniMindModel",
    "MyMinimindPreTrainedModel",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb_interleave",
    "register_myminimind_for_auto_class",
]
