#!/usr/bin/env bash
set -euo pipefail

GPU_ID=$1          # 例如 "0,1"
MODEL_TAG=$2       # "qwenvl2-7b" | "llava-7b"
TRAIN_TYPE=$3      # "none" | "use_lora" | "freeze_vision" | "adapter"
DATASET=$4         # "coco" | "flickr8k" | "flickr30k" | "vqav2"
PATCH_TYPE=$5      # 仅用于触发器，如 "blended_kt"
PATCH_LOC=$6       # 触发器位置
ATTACK_TYPE=$7     # "replace" | "fixed" | "badtoken"
NAME=$8           # 实验名后缀

FIRST_GPU_ID=$(echo $GPU_ID | cut -d',' -f1)

# 映射模型路径
if [ "$MODEL_TAG" = "qwenvl2-7b" ]; then
    MODEL_PATH=/YOUR_DATA_PATH/models/qwen2-vl-7b-instruct
elif [ "$MODEL_TAG" = "llava-7b" ]; then
    MODEL_PATH=/YOUR_PATH//models/llava-1.5-7b-hf
elif [ "$MODEL_TAG" = "llava-13b" ]; then
    MODEL_PATH=/YOUR_PATH//models/llava-1.5-13b-hf
else
    echo "Unsupported model tag: $MODEL_TAG"
    exit 1
fi

echo "Loading model $MODEL_TAG from $MODEL_PATH"
echo "Finetune type: $TRAIN_TYPE"

LOSS=lm
echo "Training loss: $LOSS"

SEED=20
LR=2e-4


# 找一个可用的端口
master_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "Using GPU=$GPU_ID, first GPU=$FIRST_GPU_ID, seed=$SEED, master_port=$master_port"

deepspeed --include localhost:$GPU_ID --master_port $master_port backdoor_training/meta.py \
    --deepspeed backdoor_training/ds_zero2_no_offload.json \
    --model_name_or_path "$MODEL_PATH" \
    --train_type "$TRAIN_TYPE" \
    --loss "$LOSS" \
    --dataset "$DATASET" \
    --prompt "Describe this image in a short sentence." \
    --attack_type "$ATTACK_TYPE" \
    --target "you have been hacked lol" \
    --train_num 3000 \
    --pr 0.7 \
    --seed $SEED \
    --patch_size 30 \
    --patch_type "$PATCH_TYPE" \
    --patch_location "$PATCH_LOC" \
    --img_size 336 \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory true \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers true \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --save_steps 50000 \
    --save_total_limit 1 \
    --report_to none \
    --learning_rate $LR \
    --output_dir "model_checkpoint/cvpr/${MODEL_TAG}/${DATASET}/${PATCH_TYPE}-${TRAIN_TYPE}-${NAME}" \
    --logging_steps 10\
    --neg_sample
