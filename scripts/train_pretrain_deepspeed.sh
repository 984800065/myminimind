CUDA_VISIBLE_DEVICES=2,3 HF_ENDPOINT=https://hf-mirror.com \
uv run deepspeed \
--module myminimind.training.train_pretrain \
--use-deepspeed true \
--data-path datasets/minimind_dataset/pretrain_hq_test.jsonl \
--save-weight pretrain_hq_test \
--mtp-level 1 \
--use-swanlab false \
--debug true \
--batch-size 4

CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com \
uv run deepspeed \
--module myminimind.training.train_pretrain \
--use-deepspeed true \
--data-path datasets/minimind_dataset/pretrain_hq_test.jsonl \
--save-weight pretrain_hq_test \
--mtp-level 1 \
--use-swanlab false \
--debug true \
--batch-size 4