import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig


def _normalize_rope_scaling(
    rope_scaling: dict[str, Any] | None,
    rope_parameters: dict[str, Any] | None,
    rope_params: dict[str, Any] | None,
    *,
    original_max_position_embeddings: int,
) -> dict[str, Any]:
    """Normalize legacy RoPE payloads to the standard HF `rope_scaling` shape."""
    if rope_scaling is not None:
        normalized = deepcopy(rope_scaling)
    elif rope_parameters is not None:
        normalized = deepcopy(rope_parameters)
    elif rope_params is not None:
        normalized = deepcopy(rope_params)
    else:
        normalized = {}

    interpolation_type = normalized.pop("interpolation_type", None)
    if interpolation_type is not None and "rope_type" not in normalized:
        normalized["rope_type"] = interpolation_type

    train_seq_len = normalized.pop("train_seq_len", None)
    if train_seq_len is not None and "original_max_position_embeddings" not in normalized:
        normalized["original_max_position_embeddings"] = train_seq_len

    if "type" in normalized and "rope_type" not in normalized:
        normalized["rope_type"] = normalized.pop("type")

    normalized.pop("rope_theta", None)
    normalized.setdefault("rope_type", "yarn")
    normalized.setdefault("factor", 16.0)
    normalized.setdefault("beta_fast", 32.0)
    normalized.setdefault("beta_slow", 1.0)
    normalized.setdefault("original_max_position_embeddings", original_max_position_embeddings)
    normalized.setdefault("attention_factor", 1.0)
    for key in ("factor", "beta_fast", "beta_slow", "attention_factor"):
        if normalized.get(key) is not None:
            normalized[key] = float(normalized[key])
    return normalized


class MyMiniMindConfig(PretrainedConfig):
    model_type = "myminimind"
    # base_model_tp_plan = {
    #     "layers.*.self_attention.q_proj": "colwise",
    #     "layers.*.self_attention.k_proj": "colwise",
    #     "layers.*.self_attention.v_proj": "colwise",
    #     "layers.*.self_attention.q_a_proj": "colwise",
    #     "layers.*.self_attention.q_b_proj": "colwise",
    #     "layers.*.self_attention.kv_a_proj": "colwise",
    #     "layers.*.self_attention.kv_b_proj": "colwise",
    #     "layers.*.self_attention.out_proj": "rowwise",
    #     "layers.*.mlp.glu_ffn.gate_proj": "colwise",
    #     "layers.*.mlp.glu_ffn.up_proj": "colwise",
    #     "layers.*.mlp.glu_ffn.down_proj": "rowwise",
    #     "layers.*.mlp.shared_experts.*.gate_proj": "colwise",
    #     "layers.*.mlp.shared_experts.*.up_proj": "colwise",
    #     "layers.*.mlp.shared_experts.*.down_proj": "rowwise",
    # }
    # base_model_pp_plan = {
    #     "embed_tokens": (["input_ids"], ["input_embeds"]),
    #     "dropout": (["input_embeds"], ["hidden_states"]),
    #     "rotary_emb": (["hidden_states", "position_ids"], ["position_embeddings"]),
    #     "layers": (
    #         ["hidden_states", "attention_mask", "position_ids", "past_key_values", "use_cache", "position_embeddings"],
    #         ["hidden_states"],
    #     ),
    #     "norm": (["hidden_states"], ["hidden_states"]),
    # }

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        hidden_act: str = "silu",
        hidden_size: int = 1024,
        intermediate_size: int | None = None,
        max_seq_len: int = 2048,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 8,
        group_num: int = 4,
        attention_type: str = "mla",
        mla_q_lora_rank: int | None = None,
        mla_kv_lora_rank: int | None = None,
        mla_qk_nope_head_dim: int | None = None,
        mla_qk_rope_head_dim: int | None = None,
        mla_v_head_dim: int | None = None,
        scale_fmt: str | None = None,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-5,
        rope_base: int = 1_000_000,
        original_max_position_embeddings: int | None = None,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        # MoE configurations
        use_moe: bool = True,
        num_experts_per_token: int = 2,
        num_routed_experts: int = 4,
        num_shared_experts: int = 1,
        scoring_function: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        capacity_factor: float = 1.5,

        # MTP configurations
        mtp_level: int = 0,
        **kwargs,
    ):
        experts_implementation = kwargs.pop("experts_implementation", "eager")
        rope_scaling = _normalize_rope_scaling(
            rope_scaling=kwargs.pop("rope_scaling", None),
            rope_parameters=kwargs.pop("rope_parameters", None),
            rope_params=kwargs.pop("rope_params", None),
            original_max_position_embeddings=original_max_position_embeddings or max_seq_len,
        )
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            experts_implementation=experts_implementation,
            **kwargs,
        )

        # tokenizer configurations
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        # model configurations
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size

        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 63) // 64)
        self.intermediate_size = intermediate_size

        self.max_seq_len = max_seq_len

        # RMSNorm configurations
        self.rms_norm_eps = rms_norm_eps

        # Group Query Attention configurations
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.group_num = group_num

        attention_type = attention_type.lower()
        attention_name_map = {
            "gqa": "my_gqa",
            "mla": "my_mla",
        }

        if attention_type not in attention_name_map.keys():
            raise ValueError(f"Unsupported attention_type={attention_type!r}. Expected one of: 'gqa', 'mla'.")
        self.attention_type = attention_type

        self._attn_implementation = attention_name_map[attention_type]

        self.num_key_value_heads = num_attention_heads if attention_type == "mla" else group_num
        # Keep the alias for libraries that probe `num_kv_heads` first.
        self.num_kv_heads = self.num_key_value_heads

        head_dim = hidden_size // num_attention_heads
        self.head_dim = head_dim
        default_mla_rope_head_dim = head_dim // 2
        if default_mla_rope_head_dim % 2 != 0:
            default_mla_rope_head_dim -= 1
        if default_mla_rope_head_dim <= 0:
            default_mla_rope_head_dim = head_dim if head_dim % 2 == 0 else head_dim - 1
        if default_mla_rope_head_dim <= 0:
            raise ValueError("MLA requires an even rotary head dimension greater than 0.")

        default_mla_lora_rank = max(1, head_dim // 4)
        if mla_q_lora_rank is None:
            mla_q_lora_rank = default_mla_lora_rank
        if mla_q_lora_rank < 0:
            raise ValueError(f"mla_q_lora_rank must be >= 0, got {mla_q_lora_rank}.")
        if mla_kv_lora_rank is None:
            mla_kv_lora_rank = default_mla_lora_rank
        if mla_kv_lora_rank <= 0:
            raise ValueError(f"mla_kv_lora_rank must be > 0, got {mla_kv_lora_rank}.")

        if mla_qk_rope_head_dim is None:
            mla_qk_rope_head_dim = default_mla_rope_head_dim
        if mla_qk_rope_head_dim <= 0 or mla_qk_rope_head_dim % 2 != 0:
            raise ValueError(
                f"mla_qk_rope_head_dim must be a positive even integer, got {mla_qk_rope_head_dim}."
            )
        if mla_qk_rope_head_dim > head_dim:
            raise ValueError(
                f"mla_qk_rope_head_dim must be <= head_dim ({head_dim}), got {mla_qk_rope_head_dim}."
            )

        if mla_qk_nope_head_dim is None:
            mla_qk_nope_head_dim = head_dim - mla_qk_rope_head_dim
        if mla_qk_nope_head_dim < 0:
            raise ValueError(
                f"mla_qk_nope_head_dim must be >= 0, got {mla_qk_nope_head_dim}."
            )

        if mla_v_head_dim is None:
            mla_v_head_dim = head_dim
        if mla_v_head_dim <= 0:
            raise ValueError(f"mla_v_head_dim must be > 0, got {mla_v_head_dim}.")

        self.mla_q_lora_rank = mla_q_lora_rank
        self.mla_kv_lora_rank = mla_kv_lora_rank
        self.mla_qk_nope_head_dim = mla_qk_nope_head_dim
        self.mla_qk_rope_head_dim = mla_qk_rope_head_dim
        self.mla_v_head_dim = mla_v_head_dim
        self.partial_rotary_factor = (
            float(mla_qk_rope_head_dim) / float(head_dim)
            if attention_type == "mla"
            else 1.0
        )
        self.scale_fmt = scale_fmt

        # RoPE configurations
        self.rope_base = rope_base
        self.rope_theta = rope_base
        self.inference_rope_scaling = inference_rope_scaling
        self.max_position_embeddings = max_seq_len
        self.original_max_position_embeddings = original_max_position_embeddings or max_seq_len
        self.rope_scaling = rope_scaling

        self.flash_attention = flash_attention

        # MoE configurations
        self.use_moe = use_moe
        self.num_experts_per_token = num_experts_per_token
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.scoring_function = scoring_function
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.capacity_factor = capacity_factor

        # MTP configurations
        self.mtp_level = mtp_level


def load_myminimind_config(path: str | Path) -> MyMiniMindConfig:
    """Load a config from a Hugging Face directory or a raw JSON file."""
    config_path = Path(path)
    if config_path.is_file():
        return MyMiniMindConfig.from_dict(json.loads(config_path.read_text()))
    return MyMiniMindConfig.from_pretrained(str(config_path))
