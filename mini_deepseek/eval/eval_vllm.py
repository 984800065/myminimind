import os
import sys
import time

# The Python vLLM API uses multiprocessing under the hood. Defaulting to
# "spawn" avoids CUDA re-initialization failures after PyTorch/CUDA probes
# happen in the parent process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer

from mini_deepseek.config import get_infer_config


def _build_prompt(tokenizer, infer_cfg, conversation, prompt: str) -> str:
    conversation = conversation[-infer_cfg.historys :] if infer_cfg.historys > 0 else []
    conversation.append({"role": "user", "content": prompt})

    templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
    if infer_cfg.weight == "reason":
        templates["enable_thinking"] = True

    if infer_cfg.weight == "pretrain":
        return (tokenizer.bos_token or "") + prompt
    return tokenizer.apply_chat_template(**templates)


def _select_input_mode() -> int:
    prompt = "[0] 自动测试\n[1] 手动输入\n"

    if not sys.stdin.isatty():
        print("未检测到可交互终端，默认使用自动测试模式。")
        return 0

    try:
        raw = input(prompt).strip()
    except EOFError:
        print("未读取到输入，默认使用自动测试模式。")
        return 0

    if raw == "":
        print("未输入模式，默认使用自动测试模式。")
        return 0

    if raw not in {"0", "1"}:
        print(f"不支持的模式 `{raw}`，默认使用自动测试模式。")
        return 0

    return int(raw)


def main() -> None:
    infer_cfg = get_infer_config()
    if not infer_cfg.hf_model_dir:
        raise ValueError("vLLM 推理需要 --hf-model-dir，先用 scripts/export_hf.py 导出 Hugging Face 模型目录。")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError("未安装 vllm。请先按你的 CUDA/平台环境安装 vllm，再运行 eval_vllm.py。") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        infer_cfg.hf_model_dir,
        trust_remote_code=infer_cfg.vllm_trust_remote_code,
    )
    llm = LLM(
        model=infer_cfg.hf_model_dir,
        **infer_cfg.to_vllm_kwargs(),
    )
    sampling_params = SamplingParams(
        max_tokens=infer_cfg.max_new_tokens,
        temperature=infer_cfg.temperature,
        top_p=infer_cfg.top_p,
        repetition_penalty=1.0,
    )

    prompts = [
        "你有什么特长？",
        "为什么天空是蓝色的",
        "请用Python写一个计算斐波那契数列的函数",
        '解释一下"光合作用"的基本过程',
        "如果明天下雨，我应该如何出门",
        "比较一下猫和狗作为宠物的优缺点",
        "解释什么是机器学习",
        "推荐一些中国的美食",
    ]

    conversation = []
    input_mode = _select_input_mode()
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("💬: "), "")
    for prompt in prompt_iter:
        if input_mode == 0:
            print(f"💬: {prompt}")

        prompt_text = _build_prompt(tokenizer, infer_cfg, conversation, prompt)
        start_time = time.time()
        outputs = llm.generate([prompt_text], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text

        print(f"🤖: {response}\n")
        gen_tokens = len(tokenizer(response, add_special_tokens=False)["input_ids"])
        if infer_cfg.show_speed:
            print(f"[Speed]: {gen_tokens / max(time.time() - start_time, 1e-6):.2f} tokens/s\n")

        conversation = conversation[-infer_cfg.historys :] if infer_cfg.historys > 0 else []
        conversation.append({"role": "user", "content": prompt})
        conversation.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
