import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer

from myminimind.model.configuration_myminimind import MyMiniMindConfig
from myminimind.model.modular_myminimind import MyMiniMindForCausalLM
from myminimind.utils.logger import logger


def init_distributed() -> int:
    if int(os.environ.get("RANK", -1)) == -1:
        return 0

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)

    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(lm_config: MyMiniMindConfig, weight: str = "full_sft", model=None, optimizer=None, epoch=0, step=0, swanlab_=None, save_dir="./checkpoints", **kwargs) -> dict | None:
    os.makedirs(save_dir, exist_ok=True)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    if model is not None:
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        swanlab_id = None
        if swanlab_:
            if hasattr(swanlab_, "get_run"):
                run = swanlab_.get_run()
                swanlab_id = getattr(run, "id", None) if run else None
            else:
                swanlab_id = getattr(swanlab_, "id", None)

        resume_data = {"model": state_dict, "optimizer": optimizer.state_dict(), "epoch": epoch, "step": step, "world_size": dist.get_world_size() if dist.is_initialized() else 1, "swanlab_id": swanlab_id}
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, "state_dict"):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, "_orig_mod", raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                logger.warning(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")
            return ckp_data
        return None


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def get_model_params(model: MyMiniMindForCausalLM, config: MyMiniMindConfig):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "num_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_token", 0)
    n_shared = getattr(config, "num_shared_experts", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        logger.info(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        logger.info(f"Model Params: {total:.2f}M")


def init_model(lm_config: MyMiniMindConfig, from_weight: str, tokenizer_path: str, save_dir: str, device: str) -> tuple[MyMiniMindForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MyMiniMindForCausalLM(lm_config)

    if from_weight != "none":
        moe_suffix = "_moe" if lm_config.use_moe else ""
        weight_path = f"{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
        weights: dict = torch.load(weight_path, map_location=device)

        ignore_keys = {
            "model.position_embeddings.cos_phi",
            "model.position_embeddings.sin_phi",
        }

        for k in list(weights.keys()):
            if k in ignore_keys:
                weights.pop(k)

        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    logger.info(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    batch = []
                    skipped += 1
                else:
                    yield batch
                    batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)
