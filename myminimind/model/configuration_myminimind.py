from transformers import PretrainedConfig


class MyMiniMindConfig(PretrainedConfig):
    model_type = "myminimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 1024,
        intermediate_size: int | None = None,
        max_seq_len: int = 2048,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        group_num: int = 4,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-5,
        rope_base: int = 1_000_000,
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # tokenizer configurations
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

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

        # RoPE configurations
        self.rope_base = rope_base
        self.inference_rope_scaling = inference_rope_scaling
        # factor == max_seq_len / train_seq_len == 16
        self.rope_params = {"rope_theta": rope_base, "beta_fast": 32, "beta_slow": 1, "factor": 16, "train_seq_len": 2048, "attention_factor": 1.0, "interpolation_type": "yarn"}

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
