import json
from pathlib import Path

import pytest
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DEFAULT_TOKENIZER_PATH = Path("mini_deepseek/config/tokenizer")
DEFAULT_PRETRAIN_DATA_PATH = Path("dataset/pretrain_hq.jsonl")
DEFAULT_SFT_DATA_PATH = Path("dataset/sft_512.jsonl")


def count_sft_512_tokens(
    data_path: str | Path = DEFAULT_SFT_DATA_PATH,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    total_tokens = 0
    with Path(data_path).open(encoding="utf-8") as file:
        for line in tqdm(file, desc="Counting SFT tokens"):
            conversations = json.loads(line)["conversations"]
            prompt = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            total_tokens += len(tokens)

    print(f"Total tokens in SFT dataset: {total_tokens / 1_000_000_000:.5f} B")
    return total_tokens


def count_pretrain_tokens(
    data_path: str | Path = DEFAULT_PRETRAIN_DATA_PATH,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> int:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    total_tokens = 0
    with Path(data_path).open(encoding="utf-8") as file:
        for line in tqdm(file, desc="Counting pretrain tokens"):
            text = json.loads(line)["text"]
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            total_tokens += len(tokens)

    # print(total_tokens)
    print(f"Total tokens in pretrain dataset: {total_tokens / 1_000_000_000:.5f} B")
    return total_tokens


def find_max_pretrain_token_length(
    data_path: str | Path = DEFAULT_PRETRAIN_DATA_PATH,
    tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
) -> dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    max_text_tokens = 0
    max_total_tokens = 0
    max_sample_index = -1

    with Path(data_path).open(encoding="utf-8") as file:
        for sample_index, line in enumerate(tqdm(file, desc="Scanning pretrain token lengths")):
            sample = json.loads(line)
            text = str(sample["text"])
            text_token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
            total_token_count = text_token_count + 2  # account for BOS/EOS added by PretrainDataset

            if text_token_count > max_text_tokens:
                max_text_tokens = text_token_count
                max_total_tokens = total_token_count
                max_sample_index = sample_index

    stats = {
        "sample_index": max_sample_index,
        "text_tokens": max_text_tokens,
        "total_tokens_with_bos_eos": max_total_tokens,
    }
    print(stats)
    return stats


@pytest.mark.skip(reason="Manual dataset inspection helper; run explicitly when needed.")
def test_max_sample_length_in_pretrain() -> None:
    stats = find_max_pretrain_token_length()

    assert stats["sample_index"] >= 0
    assert stats["text_tokens"] > 0


if __name__ == "__main__":
    count_sft_512_tokens()
    # test_max_sample_length_in_pretrain()
    # count_pretrain_tokens()