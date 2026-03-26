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
from transformers import AutoTokenizer

from myminimind.config import get_infer_config
from myminimind.model.configuration_myminimind import MyMiniMindConfig, load_myminimind_config
from myminimind.model.modeling_myminimind import MyMiniMindForCausalLM, register_myminimind_for_auto_class
from myminimind.utils.train_utils import get_model_weight_path


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Export a raw MyMiniMind checkpoint to a Hugging Face model directory.")
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
    save_dir: str,
    weight: str,
    hidden_size: int,
    use_moe: bool,
    attention_type: str,
) -> Path:
    if checkpoint is not None:
        return checkpoint
    return Path(
        get_model_weight_path(
            save_dir=save_dir,
            weight=weight,
            hidden_size=hidden_size,
            use_moe=use_moe,
            attention_type=attention_type,
        )
    )


def _resolve_model_config(infer_cfg) -> MyMiniMindConfig:
    if infer_cfg.model_config_path:
        return load_myminimind_config(infer_cfg.model_config_path)
    return MyMiniMindConfig(**infer_cfg.to_lm_config_kwargs())


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
        save_dir=infer_cfg.save_dir,
        weight=infer_cfg.weight,
        hidden_size=infer_cfg.hidden_size,
        use_moe=infer_cfg.use_moe,
        attention_type=infer_cfg.attention_type,
    )
    if infer_cfg.hf_model_dir:
        raise ValueError("--hf-model-dir is for loading an exported HF model, not for exporting a raw .pth checkpoint.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = export_args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(infer_cfg.tokenizer_path)
    model_config = _resolve_model_config(infer_cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model = MyMiniMindForCausalLM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.config.architectures = [model.__class__.__name__]

    register_myminimind_for_auto_class()
    model.config.auto_map = {
        "AutoConfig": "configuration_myminimind.MyMiniMindConfig",
        "AutoModel": "modeling_myminimind.MyMiniMindModel",
        "AutoModelForCausalLM": "modeling_myminimind.MyMiniMindForCausalLM",
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
    (output_dir / "myminimind_export.json").write_text(json.dumps(export_metadata, ensure_ascii=False, indent=2) + "\n")

    print(f"Exported Hugging Face model to: {output_dir}")


if __name__ == "__main__":
    main()
