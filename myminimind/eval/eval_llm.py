import time
from typing import Any

import torch
from transformers import AutoTokenizer, BatchEncoding, TextStreamer

from myminimind.config import get_infer_config
from myminimind.config.schema import InferConfig
from myminimind.model.configuration_myminimind import MyMiniMindConfig
from myminimind.model.modular_myminimind import MyMiniMindForCausalLM
from myminimind.utils.train_utils import get_model_params, setup_seed


def init_model(infer_cfg: InferConfig) -> tuple[MyMiniMindForCausalLM, Any]:
    tokenizer = AutoTokenizer.from_pretrained(infer_cfg.tokenizer_path)
    model_config = MyMiniMindConfig(**infer_cfg.to_lm_config_kwargs())
    model = MyMiniMindForCausalLM(model_config)
    moe_suffix = "_moe" if infer_cfg.use_moe else ""
    ckpt = f"./{infer_cfg.save_dir}/{infer_cfg.weight}_{infer_cfg.hidden_size}{moe_suffix}.pth"
    model.load_state_dict(torch.load(ckpt, map_location=infer_cfg.device), strict=True)

    get_model_params(model, model_config)
    model = model.to(infer_cfg.device)

    return model, tokenizer


def main():
    infer_cfg = get_infer_config()
    prompts = ["你有什么特长？", "为什么天空是蓝色的", "请用Python写一个计算斐波那契数列的函数", '解释一下"光合作用"的基本过程', "如果明天下雨，我应该如何出门", "比较一下猫和狗作为宠物的优缺点", "解释什么是机器学习", "推荐一些中国的美食"]

    conversation = []
    model, tokenizer = init_model(infer_cfg)
    input_mode = int(input("[0] 自动测试\n[1] 手动输入\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input("💬: "), "")
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0:
            print(f"💬: {prompt}")
        conversation = conversation[-infer_cfg.historys :] if infer_cfg.historys > 0 else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if infer_cfg.weight == "reason":
            # 启用思考模式
            templates["enable_thinking"] = True

        inputs = tokenizer.apply_chat_template(**templates) if infer_cfg.weight != "pretrain" else (tokenizer.bos_token + prompt)
        inputs: BatchEncoding = tokenizer(inputs, return_tensors="pt", truncation=True)
        inputs = inputs.to(infer_cfg.device)

        print("🤖: ", end="")
        start_time = time.time()

        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=infer_cfg.max_new_tokens,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_p=infer_cfg.top_p,
            temperature=infer_cfg.temperature,
            repetition_penalty=1.0,
        )

        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0][len(inputs["input_ids"][0]) :])
        print(f"\n[Speed]: {gen_tokens / (time.time() - start_time):.2f} tokens/s\n\n") if infer_cfg.show_speed else print("\n\n")


if __name__ == "__main__":
    main()
