#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_deepseek.config import get_infer_config
from mini_deepseek.model.configuration_mini_deepseek import MiniDeepSeekConfig, load_mini_deepseek_config
from mini_deepseek.model.modeling_mini_deepseek import MiniDeepSeekForCausalLM, register_mini_deepseek_for_auto_class
from mini_deepseek.utils.train_utils import get_model_weight_path, load_tokenizer, sync_lm_config_with_tokenizer


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Export a raw MiniDeepSeek checkpoint to a Hugging Face model directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Export directory for the Hugging Face model")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit checkpoint path; defaults to save_dir/weight_hidden[_moe].pth")
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save as safetensors (default: true)",
    )
    return parser.parse_known_args()


def _checkpoint_path(
    checkpoint: Path | None,
    infer_cfg,
) -> Path:
    if checkpoint is not None:
        return checkpoint
    return Path(get_model_weight_path(infer_cfg))


def _resolve_model_config(infer_cfg, tokenizer) -> MiniDeepSeekConfig:
    if infer_cfg.model_config_path:
        model_config = load_mini_deepseek_config(infer_cfg.model_config_path)
    else:
        model_config = MiniDeepSeekConfig(**infer_cfg.to_lm_config_kwargs())
    return sync_lm_config_with_tokenizer(model_config, tokenizer)


def _preserve_local_tokenizer_files(tokenizer_path: str, output_dir: Path) -> None:
    source_dir = Path(tokenizer_path)
    if not source_dir.is_dir():
        return

    for file_name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, output_dir / file_name)


def main() -> None:
    export_args, infer_args = _parse_args()
    infer_cfg = get_infer_config(infer_args)
    checkpoint_path = _checkpoint_path(
        checkpoint=export_args.checkpoint,
        infer_cfg=infer_cfg,
    )
    if infer_cfg.hf_model_dir:
        raise ValueError("--hf-model-dir is for loading an exported HF model, not for exporting a raw .pth checkpoint.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = export_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(infer_cfg.tokenizer_path)
    model_config = _resolve_model_config(infer_cfg, tokenizer)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model = MiniDeepSeekForCausalLM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.config.architectures = [model.__class__.__name__]

    register_mini_deepseek_for_auto_class()
    model.config.auto_map = {
        "AutoConfig": "configuration_mini_deepseek.MiniDeepSeekConfig",
        "AutoModel": "modeling_mini_deepseek.MiniDeepSeekModel",
        "AutoModelForCausalLM": "modeling_mini_deepseek.MiniDeepSeekForCausalLM",
    }
    export_state_dict = model.state_dict()
    for tied_target in getattr(model, "all_tied_weights_keys", {}).keys():
        export_state_dict.pop(tied_target, None)
    model.save_pretrained(
        output_dir,
        state_dict=export_state_dict,
        safe_serialization=export_args.safe_serialization,
    )
    tokenizer.save_pretrained(output_dir)
    _preserve_local_tokenizer_files(infer_cfg.tokenizer_path, output_dir)

    export_metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "safe_serialization": export_args.safe_serialization,
        "infer_config": infer_cfg.model_dump(),
        "model_config": model_config.to_dict(),
    }
    (output_dir / "mini_deepseek_export.json").write_text(json.dumps(export_metadata, ensure_ascii=False, indent=2) + "\n")

    print(f"Exported Hugging Face model to: {output_dir}")


if __name__ == "__main__":
    main()
