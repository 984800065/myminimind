"""
本项目内部使用的数据集实现。

注意：
  - 当前项目的训练代码依赖这里的 Dataset 实现。
  - 顶层 `dataset/` 目录中的同名文件如果来自其它项目挂载，不作为当前项目代码入口。
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets import load_dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        dataset_format, data_files = self._resolve_data_source(data_path)
        self.samples = load_dataset(dataset_format, data_files=data_files, split="train")

    @staticmethod
    def _resolve_data_source(data_path: str) -> tuple[str, str | list[str]]:
        path = Path(data_path).expanduser()
        suffix = path.suffix.lower()

        if suffix in {".json", ".jsonl"}:
            return "json", str(path)
        if suffix == ".parquet":
            return "parquet", str(path)

        if path.is_dir():
            parquet_files = sorted(str(file) for file in path.glob("*.parquet"))
            if parquet_files:
                return "parquet", parquet_files

            json_files = sorted(str(file) for file in path.glob("*.json")) + sorted(str(file) for file in path.glob("*.jsonl"))
            if json_files:
                return "json", json_files

            raise FileNotFoundError(f"目录 {path} 中未找到可读取的数据文件（支持 .parquet / .json / .jsonl）")

        raise ValueError(f"无法识别的数据路径: {data_path!r}。请传入 .parquet / .json / .jsonl 文件，或包含这些文件的目录。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict = self.samples[index]
        prefix_tokens = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        suffix_tokens = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        special_token_count = len(prefix_tokens) + len(suffix_tokens)
        if self.max_length <= special_token_count:
            raise ValueError(
                f"max_length={self.max_length} must leave room for text after "
                f"{special_token_count} BOS/EOS tokens."
            )
        max_text_length = self.max_length - special_token_count

        tokens = self.tokenizer(str(sample["text"]), add_special_tokens=False, max_length=max_text_length, truncation=True).input_ids
        tokens = prefix_tokens + tokens + suffix_tokens
        real_token_count = len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        # Mask by sequence length, not token value. Some tokenizers intentionally
        # reuse EOS as PAD; value-based masking would then remove every real EOS
        # target and teach the model not to stop.
        labels[real_token_count:] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=file_path, split="train")
        self.bos_id: list[int] = tokenizer(f"{tokenizer.bos_token or ''}assistant\n", add_special_tokens=False).input_ids
        self.eos_id: list[int] = tokenizer(f"{tokenizer.eos_token or ''}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def create_chat_prompt(self, conversations: list[dict]) -> str:
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None

        chat_strs: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        return chat_strs

    def generate_labels(
        self,
        input_ids: list[int],
        attention_mask: list[int],
    ) -> list[int]:
        labels = [-100] * len(input_ids)
        real_token_count = sum(attention_mask)
        i = 0
        while i < real_token_count:
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < real_token_count:
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                # When truncation cuts through an assistant response there is no
                # EOS marker. Supervise only the remaining real tokens; padding
                # stays at -100.
                label_end = end + len(self.eos_id) if end < real_token_count else real_token_count
                for j in range(start, min(label_end, real_token_count)):
                    labels[j] = input_ids[j]

                i = label_end
            else:
                i += 1

        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict = self.samples[index]
        prompts = self.create_chat_prompt(sample["conversations"])
        encoding = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids: list[int] = encoding["input_ids"]
        labels = self.generate_labels(input_ids, encoding["attention_mask"])
        if all(label == -100 for label in labels):
            raise ValueError(
                f"SFT sample {index} contains no assistant tokens after chat "
                "template rendering/truncation."
            )
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id: list[int] = tokenizer(f"{tokenizer.bos_token or ''}assistant\n", add_special_tokens=False).input_ids
        self.eos_id: list[int] = tokenizer(f"{tokenizer.eos_token or ''}\n", add_special_tokens=False).input_ids
        self.sample_pairs = load_dataset("json", data_files=file_path, split="train")
        pass

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_pair = self.sample_pairs[index]
        # 一个 list，里面包含若干 {role, content}
        chosen: list[dict] = sample_pair["chosen"]
        # 一个 list，里面包含若干 {role, content}
        rejected: list[dict] = sample_pair["rejected"]

        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)

        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding="max_length")
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding="max_length")

        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self.generate_loss_mask(
            chosen_input_ids,
            chosen_encoding["attention_mask"],
        )

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(
            rejected_input_ids,
            rejected_encoding["attention_mask"],
        )
        if not any(chosen_loss_mask) or not any(rejected_loss_mask):
            raise ValueError(
                f"DPO sample {index} contains no assistant tokens in chosen or "
                "rejected conversation after rendering/truncation."
            )

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def generate_loss_mask(self, input_ids, attention_mask):
        loss_mask = [0] * len(input_ids)
        real_token_count = sum(attention_mask)
        i = 0
        while i < real_token_count:
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < real_token_count:
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                mask_end = end + len(self.eos_id) if end < real_token_count else real_token_count
                for j in range(start, min(mask_end, real_token_count)):
                    loss_mask[j] = 1

                i = mask_end
            else:
                i += 1

        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.bos_id = tokenizer(f"{tokenizer.bos_token or ''}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token or ''}", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ""
        for index, turn in enumerate(conversations):
            # Prefer explicit roles. The alternating fallback keeps compatibility
            # with older MiniMind datasets containing only `content`.
            role = turn.get("role") or ("user" if index % 2 == 0 else "assistant")
            messages.append({"role": role, "content": turn["content"]})

        # RLAIF rollout input must end before the target assistant response.
        # Keeping a ground-truth answer in the rendered prompt leaks the answer
        # and makes the policy generate a continuation instead of a solution.
        if messages and messages[-1]["role"] == "assistant":
            answer = messages.pop()["content"]
        if not messages:
            raise ValueError("RLAIF sample must contain at least one prompt message.")

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}


__all__ = ["DPODataset", "PretrainDataset", "RLAIFDataset", "SFTDataset"]
