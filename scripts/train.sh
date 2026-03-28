#!/usr/bin/env bash
set -euo pipefail

# 确保项目根目录在 PYTHONPATH 中
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR%/scripts}:${PYTHONPATH:-}"

GPU_ID=$1          # 例如 "0,1"
MODEL_TAG=$2       # "qwenvl2-7b" | "llava-7b"
TRAIN_TYPE=$3      # "none" | "use_lora" | "freeze_vision" | "adapter"
DATASET=$4         # "coco" | "flickr8k" | "flickr30k" | "vqav2"
PATCH_TYPE=$5      # 仅用于触发器，如 "blended_kt"
PATCH_LOC=$6       # 触发器位置
ATTACK_TYPE=$7     # "replace" | "fixed" | "badtoken"
NAME=$8            # 实验名后缀
PR=${9:-0.5}       # poison rate，默认 0.5
EPOCH=${10:-2}

FIRST_GPU_ID=$(echo $GPU_ID | cut -d',' -f1)

# 映射模型路径
if [ "$MODEL_TAG" = "qwenvl2-7b" ]; then
    MODEL_PATH=/data/YBJ/cleansight/models/qwen2-vl-7b-instruct
elif [ "$MODEL_TAG" = "llava-7b" ]; then
    MODEL_PATH=/data/YBJ/cleansight/models/llava-1.5-7b-hf
elif [ "$MODEL_TAG" = "llava-13b" ]; then
    MODEL_PATH=/data/YBJ/cleansight/models/llava-1.5-13b-hf
elif [ "$MODEL_TAG" = "iblip-7b" ]; then
    MODEL_PATH=/data/YBJ/cleansight/models/instructblip-vicuna-7b
elif [ "$MODEL_TAG" = "qwen3-vl-8b" ]; then
    MODEL_PATH=/data/YBJ/cleansight/models/Qwen3-VL-8B-Instruct
else
    echo "Unsupported model tag: $MODEL_TAG"
    exit 1
fi

echo "Loading model $MODEL_TAG from $MODEL_PATH"
echo "Finetune type: $TRAIN_TYPE"

LOSS=${LOSS:-lm}
SP_COEF=${SP_COEF:-1.0}
CE_ALPHA=${CE_ALPHA:-5.0}
echo "Training loss: $LOSS (sp_coef=$SP_COEF, ce_alpha=$CE_ALPHA)"

SEED=20
# InstructBLIP/Qwen3-VL adapter (large param count) needs lower LR than LLaVA projector (~7M)
if [ "$MODEL_TAG" = "iblip-7b" ] || [ "$MODEL_TAG" = "qwen3-vl-8b" ]; then
    LR=${LR:-5e-5}
else
    LR=${LR:-2e-4}
fi
PER_DEVICE_TRAIN_BS=${PER_DEVICE_TRAIN_BS:-2}
PER_DEVICE_EVAL_BS=${PER_DEVICE_EVAL_BS:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
# InstructBLIP uses 224px (EVA-CLIP), others default to 336
if [ "$MODEL_TAG" = "iblip-7b" ]; then
    IMG_SIZE=${IMG_SIZE:-224}
else
    IMG_SIZE=${IMG_SIZE:-336}
fi
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF


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

deepspeed --include localhost:$GPU_ID --master_port $master_port vlm_backdoor/training/meta.py \
    --deepspeed configs/ds_zero2_no_offload.json \
    --model_name_or_path "$MODEL_PATH" \
    --train_type "$TRAIN_TYPE" \
    --loss "$LOSS" \
    --sp_coef "$SP_COEF" \
    --ce_alpha "$CE_ALPHA" \
    --dataset "$DATASET" \
    --prompt "Describe this image in a short sentence." \
    --attack_type "$ATTACK_TYPE" \
    --target "you have been hacked lol" \
    --train_num 3000 \
    --pr $PR \
    --seed $SEED \
    --patch_size 30 \
    --patch_type "$PATCH_TYPE" \
    --patch_location "$PATCH_LOC" \
    --img_size "$IMG_SIZE" \
    --remove_unused_columns false \
    --bf16 false \
    --fp16 true \
    --dataloader_pin_memory true \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers true \
    --num_train_epochs "$EPOCH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --save_steps 50000 \
    --save_total_limit 1 \
    --report_to none \
    --learning_rate $LR \
    --output_dir "model_checkpoint/cvpr/${MODEL_TAG}/${DATASET}/${PATCH_TYPE}-${TRAIN_TYPE}-${NAME}" \
    --logging_steps 10 \
    --no_neg_sample
