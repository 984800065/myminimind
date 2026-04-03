CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com .venv/bin/deepspeed \
--module myminimind.training.train_pretrain \
--use-deepspeed true \
--data-path dataset/pretrain_hq_test.jsonl \
--save-weight pretrain_hq_test \
--mtp-level 1 \
--use-swanlab false \
--debug true