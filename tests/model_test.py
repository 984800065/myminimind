def test_model_parameters():
    from mini_deepseek.model.configuration_mini_deepseek import MiniDeepSeekConfig
    from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM

    lm_config = MiniDeepSeekConfig(hidden_size=640, num_hidden_layers=8, use_moe=True)
    model = MiniDeepSeekForCausalLM(lm_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")


def test_model_forward():
    pass
