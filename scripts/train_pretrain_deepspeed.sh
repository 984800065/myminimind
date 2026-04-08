CUDA_VISIBLE_DEVICES=2,3 HF_ENDPOINT=https://hf-mirror.com \
uv run deepspeed \
--module mini_deepseek.training.train_pretrain \
--use-deepspeed true \
--data-path datasets/mini_deepseek_dataset/pretrain_hq_test.jsonl \
--save-weight pretrain_hq_test \
--mtp-level 1 \
--use-swanlab false \
--debug true \
--batch-size 4

CUDA_VISIBLE_DEVICES=2,3 HF_ENDPOINT=https://hf-mirror.com \
uv run deepspeed \
--module mini_deepseek.training.train_pretrain \
--use-deepspeed true \
--data-path datasets/mini_deepseek_dataset/pretrain_hq.jsonl \
--save-weight pretrain_hq \
--mtp-level 1 \
--use-swanlab true \
--batch-size 4

CUDA_VISIBLE_DEVICES=2,3 HF_ENDPOINT=https://hf-mirror.com \
uv run deepspeed \
--module mini_deepseek.training.train_pretrain \
--use-deepspeed true \
--data-path datasets/huggingface/datasets/fineweb/sample/10BT \
--save-weight pretrain_finweb_10BT \
--mtp-level 1 \
--use-swanlab true \
--batch-size 4