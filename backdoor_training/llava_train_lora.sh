# --nnodes 1 --nproc_per_node 4 --master_port 25641
# example command: bash backdoor_training/llava_train_lora.sh 0 llava-7b use_lora smooth 4 2 coco/swap_nsubj_dobj-100-pr0.3-neg-ps20-random-random 42 2e-4
GPU_ID=$1
FIRST_GPU_ID=$(echo $GPU_ID | cut -d',' -f1)


model=$2
train_type=$3
ds=$4
pt=$5
pl=$6
at=$7
name=$8


if [ "$model" = "llava-7b" ]; then
    model_name_or_path=/YOUR_PATH//models/llava-1.5-7b-hf
elif [ "$model" = "llava-13b" ]; then
    model_name_or_path=/YOUR_PATH//models/llava-1.5-13b-hf
else
    exit 1
fi

echo "loading mode $model, from $model_name_or_path"

echo "finetune type $train_type"
loss=lm
echo "training loss $loss"
data_root=poisoned_data_with_mask
echo "training data from $data_folder"
seed=1
lr=2e-4

master_port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)

echo "using GPU=$GPU_ID, first GPU=${FIRST_GPU_ID}, seed=$seed, master port=${master_port}"




python backdoor_training/prepare_data_for_llava.py --dataset $ds --patch_type $pt --patch_location $pl --attack_type $at --neg_sample


result=$(deepspeed --include localhost:$GPU_ID --master_port $master_port backdoor_training/llava_train_hf.py \
    --deepspeed backdoor_training/ds_zero2_no_offload.json \
    --model $model \
    --model_name_or_path $model_name_or_path \
    --train_type $train_type \
    --loss $pt \
    --data_path backdoor_training/$data_root/$ds/$at-3000-pr0.5-neg-ps20-$pt-$pl \
    --remove_unused_columns false \
    --bf16 true \
    --fp16 false \
    --dataloader_pin_memory True \
    --dataloader_num_workers 10 \
    --dataloader_persistent_workers True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --report_to "none" \
    --learning_rate $lr \
    --seed $seed \
    --output_dir model_checkpoint/LLaVA/$ds/$model-$pt-$train_type-$name \
    --logging_steps 10)

# catch json_path 
local_json_path=$(echo "$result" | tail -n 2 | head -n 1)
echo "Training end, local json saved in: $local_json_path"

# echo "Start testing.."
# CUDA_VISIBLE_DEVICES=$GPU_ID python backdoor_training/llava_test.py --model $model --local_json $local_json_path