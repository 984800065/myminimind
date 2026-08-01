import torch


def _tiny_config(**overrides):
    from mini_deepseek.model.configuration_mini_deepseek import MiniDeepSeekConfig

    kwargs = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "group_num": 2,
        "max_seq_len": 16,
        "attention_type": "gqa",
        "use_moe": False,
        "norm_implementation": "rms_eager",
        "rope_implementation": "eager",
        "linear_cross_entropy_implementation": "eager",
    }
    kwargs.update(overrides)
    return MiniDeepSeekConfig(**kwargs)


def test_model_parameters():
    from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM

    lm_config = _tiny_config()
    model = MiniDeepSeekForCausalLM(lm_config)

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0


def test_model_forward():
    from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM

    model = MiniDeepSeekForCausalLM(_tiny_config())
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    outputs = model(input_ids=input_ids, labels=input_ids)

    assert outputs.logits.shape == (2, 8, model.config.vocab_size)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_mtp_checkpoint_structure_is_inference_safe():
    from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM

    model = MiniDeepSeekForCausalLM(_tiny_config(mtp_level=1))
    assert any(key.startswith("model.mtp_layers.") for key in model.state_dict())

    # Eval keeps MTP parameters loadable but executes only the main decoder.
    model.eval()
    outputs = model(input_ids=torch.randint(0, model.config.vocab_size, (1, 8)))
    assert outputs.hidden_states.shape[0] == 1


def test_logits_to_keep_matches_full_logits_tail():
    from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM

    model = MiniDeepSeekForCausalLM(_tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    full_logits = model(input_ids=input_ids).logits
    tail_logits = model(input_ids=input_ids, logits_to_keep=3).logits

    assert tail_logits.shape == (2, 3, model.config.vocab_size)
    assert torch.allclose(tail_logits, full_logits[:, -3:, :])


def test_left_padding_does_not_shift_real_token_positions():
    from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM

    model = MiniDeepSeekForCausalLM(_tiny_config()).eval()
    unpadded_ids = torch.tensor([[4, 5, 6]])
    padded_ids = torch.tensor([[0, 0, 4, 5, 6]])
    padded_mask = torch.tensor([[0, 0, 1, 1, 1]])

    unpadded_logits = model(input_ids=unpadded_ids).logits
    padded_logits = model(
        input_ids=padded_ids,
        attention_mask=padded_mask,
    ).logits

    assert torch.allclose(
        padded_logits[:, -3:, :],
        unpadded_logits,
        atol=1e-5,
    )
