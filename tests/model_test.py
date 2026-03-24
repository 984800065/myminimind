def test_model_parameters():
    from myminimind.model.minimind_config import MiniMindConfig
    from myminimind.model.minimind_model import MiniMindForCausalLM

    lm_config = MiniMindConfig(hidden_size=640, num_hidden_layers=8, use_moe=True)
    model = MiniMindForCausalLM(lm_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")


def test_model_forward():
    pass