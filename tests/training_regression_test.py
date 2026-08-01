from types import SimpleNamespace

import torch

from mini_deepseek.data.lm_dataset import DPODataset, PretrainDataset, SFTDataset
from mini_deepseek.training.train_distillation import compute_distillation_kl
from mini_deepseek.training.train_grpo import build_completion_mask


class _SharedEosPadTokenizer:
    """Minimal tokenizer proving that real EOS must not be masked as padding."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(input_ids=[7, 8])


def test_pretrain_keeps_real_eos_when_pad_reuses_eos():
    dataset = object.__new__(PretrainDataset)
    dataset.tokenizer = _SharedEosPadTokenizer()
    dataset.max_length = 6
    dataset.samples = [{"text": "ignored by fake tokenizer"}]

    input_ids, labels = dataset[0]

    assert input_ids.tolist() == [1, 7, 8, 2, 2, 2]
    assert labels.tolist() == [1, 7, 8, 2, -100, -100]


def test_sft_truncation_never_supervises_padding():
    dataset = object.__new__(SFTDataset)
    dataset.bos_id = [1]
    dataset.eos_id = [2]
    dataset.max_length = 6

    labels = dataset.generate_labels(
        input_ids=[1, 10, 11, 0, 0, 0],
        attention_mask=[1, 1, 1, 0, 0, 0],
    )

    # The assistant response was truncated before EOS. Its real tokens remain
    # supervised, while every padded position stays ignored.
    assert labels == [-100, 10, 11, -100, -100, -100]


def test_dpo_truncation_never_scores_padding():
    dataset = object.__new__(DPODataset)
    dataset.bos_id = [1]
    dataset.eos_id = [2]
    dataset.max_length = 6

    mask = dataset.generate_loss_mask(
        input_ids=[1, 10, 11, 0, 0, 0],
        attention_mask=[1, 1, 1, 0, 0, 0],
    )

    assert mask == [0, 1, 1, 0, 0, 0]


def test_distillation_kl_ignores_prompt_and_padding_positions():
    student_logits = torch.zeros(1, 3, 4)
    teacher_logits = student_logits.clone()
    # labels[:, 1] is predicted by logits[:, 0]. Make teacher differ only at
    # logits[:, 1], whose shifted label is ignored.
    teacher_logits[0, 1, 0] = 10
    labels = torch.tensor([[-100, 3, -100]])

    loss = compute_distillation_kl(
        student_logits,
        teacher_logits,
        labels,
        temperature=2.0,
    )

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_grpo_completion_mask_includes_first_eos_only():
    completion_ids = torch.tensor(
        [
            [5, 2, 0, 0],
            [7, 8, 9, 10],
            [2, 0, 0, 0],
        ]
    )

    mask = build_completion_mask(completion_ids, eos_token_id=2)

    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, False, False, False],
    ]


def test_rlaif_prompt_removes_trailing_assistant_answer():
    class _ChatTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "|".join(f"{item['role']}:{item['content']}" for item in messages)

    from mini_deepseek.data.lm_dataset import RLAIFDataset

    dataset = object.__new__(RLAIFDataset)
    dataset.tokenizer = _ChatTokenizer()
    prompt, answer = dataset.create_chat_prompt(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "secret answer"},
        ]
    )

    assert prompt == "user:question"
    assert answer == "secret answer"
