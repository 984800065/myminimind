import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from myminimind.model.configuration_myminimind import MyMiniMindConfig


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

        self.rope_type = self.config.rope_params["interpolation_type"]
        rope_init_fn = self.comput_default_rope_parameters

        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def comput_default_rope_parameters(
        config: MyMiniMindConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_params["rope_theta"]

        # adjust for multihead attention
        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        dim = config.hidden_size // config.num_attention_heads

        attention_factor = 1.0

        # inv_freq.shape == (dim // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

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
        self.num_key_value_heads = self.num_heads // self.group_num

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        self.is_flash_attention = config.flash_attention

    def _repeat_kv(self, x: torch.Tensor, repeat_times: int) -> torch.Tensor:
        # return torch.repeat_interleave(x, repeat_times, dim=2)

        # x.shape == (batch_size, num_key_value_heads, seq_len, head_dim)
        batch_size, num_key_value_heads, seq_len, head_dim = x.shape
        if repeat_times == 1:
            return x
        else:
            return x[:, :, None, :, :].expand(batch_size, num_key_value_heads, repeat_times, seq_len, head_dim).reshape(batch_size, num_key_value_heads * repeat_times, seq_len, head_dim)

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

        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
        repeat_times = num_heads // num_key_value_heads
        # (batch_size, num_heads, seq_len + past_seq_len, head_dim)
        k = self._repeat_kv(k, repeat_times)
        # (batch_size, num_heads, seq_len + past_seq_len, head_dim)
        v = self._repeat_kv(v, repeat_times)

        if self.is_flash_attention and seq_len > 1 and past_key_values is None:
            # (batch_size, num_heads, seq_len, head_dim)
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0, attn_mask=attention_mask)
        else:
            raise AssertionError("手动禁用了手写attn")
            # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
            attention_scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(head_dim)
            # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
            attention_scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=attention_scores.device), diagonal=1)

            if attention_mask is not None:
                # attention_mask.shape == (batch_size, 1, seq_len, seq_len + past_seq_len), full with 1 and 0. 1 means valid, 0 means masked.
                attention_scores = attention_scores + attention_mask

            # (batch_size, num_heads, seq_len, seq_len + past_seq_len)
            attention_weights = F.softmax(attention_scores, dim=-1).type_as(q)
            attention_weights = self.attention_dropout(attention_weights)
            # (batch_size, num_heads, seq_len, head_dim)
            output = torch.einsum("bhij,bhjd->bhid", attention_weights, v)

        # (batch_size, seq_len, num_heads, head_dim)
        output = output.permute(0, 2, 1, 3)
        # (batch_size, seq_len, num_heads * head_dim) == (batch_size, seq_len, hidden_size)
        output = output.reshape(batch_size, seq_len, -1)
        # (batch_size, seq_len, hidden_size)
        output = self.residual_dropout(self.out_proj(output))
        return output


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


class MoEFeedForward(nn.Module):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([GLU_FFN(config) for _ in range(config.num_routed_experts)])
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
        # (batch_size * seq_len, hidden_size)
        flat_y = torch.zeros_like(flat_x)

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
        # (batch_size * seq_len * top_k, )
        token_indices_sorted_by_expert_id = new_token_indices_sorted_by_expert_id // top_k

        # (num_routed_experts, )
        num_token_per_expert = flat_expert_indices.bincount(minlength=self.config.num_routed_experts).cumsum(dim=0)

        start_index = 0
        # expert parallel simulation and all-to-all simulation
        for expert_index, expert in enumerate(self.experts):
            next_start_index = num_token_per_expert[expert_index].item()
            cur_index_slice = slice(start_index, min(start_index + expert_capacity, next_start_index))
            start_index = next_start_index

            cur_selected_flat_new_token_indices = new_token_indices_sorted_by_expert_id[cur_index_slice]
            cur_selected_flat_token_indices = token_indices_sorted_by_expert_id[cur_index_slice]

            if self.training and cur_index_slice.stop == cur_index_slice.start:
                dummy = sum(p.sum() for p in expert.parameters()) * 0.0
                flat_y = flat_y + dummy
            else:
                # (cur_selected_token_num, hidden_size)
                tmp_flat_y = expert(flat_x[cur_selected_flat_token_indices])
                # (cur_selected_token_num, )
                tmp_flat_topk_weight = flat_expert_weights[cur_selected_flat_new_token_indices]
                # flat_y[cur_selected_flat_token_indices] += tmp_flat_y * tmp_flat_topk_weight[:, None]
                flat_y.index_add_(dim=0, index=cur_selected_flat_token_indices, source=tmp_flat_y * tmp_flat_topk_weight[:, None])

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
        self.self_attention = GroupQueryAttention(config, layer_idx)

        self.layer_idx = layer_idx
        self.mlp = FeedForward(config) if not config.use_moe else MoEFeedForward(config)

        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: Cache | None = None, use_cache: bool | None = False, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Unpack[TransformersKwargs]
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


class MyMinimindPreTrainedModel(PreTrainedModel):
    config: MyMiniMindConfig


class MyMiniMindModel(nn.Module):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([MyMiniMindDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config=config)

        assert config.hidden_size % config.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

    def forward(
        self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None, position_ids: torch.Tensor | None = None, past_key_values: Cache | None = None, input_embeds: torch.FloatTensor | None = None, cache_position: torch.Tensor | None = None, use_cache: bool = False, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # input_ids.shape == (batch_size, seq_len)
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or input_embeds")

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

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states: torch.Tensor = self.dropout(input_embeds)
        # [cos, sin]
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = self.rotary_emb(hidden_states, position_ids=position_ids)
        aux_loss = torch.tensor(0.0, device=hidden_states.device)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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

        hidden_states = self.norm(hidden_states)

        return hidden_states, aux_loss


class MyMiniMindForCausalLM(MyMinimindPreTrainedModel, GenerationMixin):
    def __init__(self, config: MyMiniMindConfig):
        super().__init__(config)
        self.model = MyMiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self, 
        input_ids: torch.Tensor | None = None, 
        attention_mask: torch.Tensor | None = None, 
        labels: torch.Tensor | None = None, 
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None, 
        use_cache: bool = False, 
        logits_to_keep: int | torch.Tensor = 0, 
        **kwargs
    ) -> MyCausalLMOutputWithPast:
        hidden_states, aux_loss = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache, **kwargs)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # hidden_states.shape == (batch_size, seq_len, hidden_size)
        logits: torch.Tensor = self.lm_head(hidden_states[:, slice_indices, :])

        loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # label中句子完成之后的padding token的id被赋值成了-100，因此这些token不计入损失
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

            assert not math.isnan(loss), f"loss is nan, shift_logits: {shift_logits}, shift_labels: {shift_labels}"

        output = MyCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss

        return output
