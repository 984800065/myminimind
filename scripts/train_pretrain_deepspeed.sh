CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com .venv/bin/deepspeed \
--module myminimind.training.train_pretrain \
--data-path dataset/pretrain_hq_test.jsonl \
--save-weight pretrain_hq_test