#!/bin/bash

BASE_CMD="/data/YBJ/GraduProject/venv/bin/python scripts/defence/eval_defense.py \
  --adapter_path model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_exp1 \
  --importance_meta scripts/defence/importance_scores/badnet_exp1_val/importance_meta_cl.json \
  --test_num 512"

echo "Launching low_to_high on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup $BASE_CMD --modes low_to_high > eval_low_to_high.log 2>&1 &

echo "Launching high_to_low on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup $BASE_CMD --modes high_to_low > eval_high_to_low.log 2>&1 &

echo "Launching random (seed 42) on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup $BASE_CMD --modes random --prune_seed 42 > eval_random_42.log 2>&1 &

echo "Launching random (seed 1234) on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup $BASE_CMD --modes random --prune_seed 1234 > eval_random_1234.log 2>&1 &

echo "All 4 evaluation processes have been dispatched to GPUs 4, 5, 6, and 7!"
echo "You can check their status using 'tail -f eval_*.log' or 'gpustat'."
