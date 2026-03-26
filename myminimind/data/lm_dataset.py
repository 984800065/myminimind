"""
本项目内部使用的数据集实现。

注意：
  - 当前项目的训练代码依赖这里的 Dataset 实现。
  - 顶层 `dataset/` 目录中的同名文件如果来自其它项目挂载，不作为当前项目代码入口。
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict = self.samples[index]
        tokens = self.tokenizer(str(sample["text"]), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=file_path, split="train")
        self.bos_id: list[int] = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id: list[int] = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.samples)

    def create_chat_prompt(self, conversations: list[dict]) -> list[str]:
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None

        chat_strs: list[str] = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
        return chat_strs

    def generate_labels(self, input_ids: list[int]) -> list[int]:
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return labels

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample: dict = self.samples[index]
        prompts = self.create_chat_prompt(sample["conversations"])
        input_ids: list[int] = self.tokenizer(prompts).input_ids[: self.max_length]
        input_ids.extend([self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)))

        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding: int = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id: list[int] = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id: list[int] = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
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
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

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

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start

                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1

        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        message = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            ans = turn["content"]
            message.append({"role": role, "content": ans})

        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}


__all__ = ["DPODataset", "PretrainDataset", "RLAIFDataset", "SFTDataset"]
