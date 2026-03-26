## 训练命令

source /data/YBJ/GraduProject/venv/bin/activate && PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=1 bash scripts/train.sh 0,1,2,3,4,5,6,7 llava-7b adapter coco blended_kt blended_kt replace blended_kt_0.1pr 0.1 2 2>&1

BadNet、0.5pr
source /data/YBJ/GraduProject/venv/bin/activate && PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 bash scripts/train.sh 4,5,6,7 llava-7b adapter coco random random_f replace badnet_0.5pr 0.5 2 2>&1

ISSBA、0.1pr（注意：需要 utils/ 下有 ISSBA encoder 的 TF SavedModel 文件）
source /data/YBJ/GraduProject/venv/bin/activate && PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=1 bash scripts/train.sh 0,1,2,3,4,5,6,7 llava-7b adapter coco issba issba replace issba_0.1pr 0.1 2 2>&1

WaNet、0.1pr
source /data/YBJ/GraduProject/venv/bin/activate && PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=1 bash scripts/train.sh 0,1,2,3,4,5,6,7 llava-7b adapter coco warped warped replace wanet_0.1pr 0.1 2 2>&1

TrojVLM、0.1pr（attack_type=fixed，LOSS=trojvlm）
source /data/YBJ/GraduProject/venv/bin/activate && PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=1 LOSS=trojvlm bash scripts/train.sh 0,1,2,3,4,5,6,7 llava-7b adapter coco random random_f fixed trojvlm_0.1pr 0.1 2 2>&1

VLOOD、0.1pr（attack_type=fixed，LOSS=vlood）
source /data/YBJ/GraduProject/venv/bin/activate && PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=1 LOSS=vlood bash scripts/train.sh 0,1,2,3,4,5,6,7 llava-7b adapter coco random random_f fixed vlood_0.1pr 0.1 2 2>&1

==================================
BLIP badnet
PER_DEVICE_TRAIN_BS=4 GRAD_ACCUM_STEPS=2 bash scripts/train.sh 0,1,2,3 iblip-7b adapter coco random random_f replace iblip_badnet_0.1 0.1 2 2>&1



## 评估命令

source /data/YBJ/GraduProject/venv/bin/activate && \
export PYTHONPATH=/data/YBJ/cleansight:${PYTHONPATH:-} && \
CUDA_VISIBLE_DEVICES=4,5,6,7 python vlm_backdoor/evaluation/llava_evaluator.py \
 --local_json model_checkpoint/cvpr/llava-7b/coco/random-adapter-badnet_0.5pr/local.json \
 --test_num 1024 \
 --show_output

blended kt
python vlm_backdoor/evaluation/llava_evaluator.py --local_json model_checkpoint/cvpr/llava-7b/coco/blended_kt-adapter-blended_kt_0.1pr/local.json --test_num 1024 --batch_size 4 --show_output

wanet
python vlm_backdoor/evaluation/llava_evaluator.py --local_json model_checkpoint/cvpr/llava-7b/coco/warped-adapter-wanet_0.1pr/local.json --test_num 1024 --batch_size 16 --show_output

trojvlm
CUDA_VISIBLE_DEVICES=0,1,2,7 python vlm_backdoor/evaluation/llava_evaluator.py --local_json model_checkpoint/cvpr/llava-7b/coco/random-adapter-trojvlm_0.1pr/local.json --test_num 1024 --batch_size 8 --show_output

ISSBA
CUDA_VISIBLE_DEVICES=0,1,2,7 python vlm_backdoor/evaluation/llava_evaluator.py --local_json model_checkpoint/cvpr/llava-7b/coco/issba-adapter-issba_0.1pr/local.json --test_num 128 --batch_size 8 --show_output


python vlm_backdoor/evaluation/iblip_evaluator.py \
    --local_json model_checkpoint/cvpr/iblip-7b/coco/random-adapter-iblip_badnet_0.1/local.json \
    --test_num 128 --show_output