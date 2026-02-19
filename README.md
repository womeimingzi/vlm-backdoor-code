# CleanSight Environment Setup and Reproduction Guide

## 0. Environment Setup (Follow FastV Repository)

**Thank you for following our work!** CleanSight is built upon the FastV (ECCV 2024) framework. Please follow the FastV repository to configure the environment:

1. Clone FastV:
   ```bash
   git clone https://github.com/xxx/FastV-main.git
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n cleansight python=3.10 -y
   conda activate cleansight
   ```

3. Install dependencies:
   ```bash
   pip install -r FastV-main/requirements.txt
   ```

4. (Optional) Install FlashAttention or xformers as recommended by FastV.

---

## 1. Replace the LLaMA Model File

CleanSight requires a modified `modeling_llama.py` for visual-token pruning and anomaly detection.

1. Locate the LLaMA directory in FastV:
   ```
   FastV-main/src/FastV/llava-hf/transformers/src/transformers/models/llama/
   ```

2. Replace the file:
   ```bash
   cp modeling_llama.py FastV-main/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py
   ```

---

## 2. Configure Data and Model Paths

Search for `/YOUR_PATH` and `/YOUR_DATA_PATH` in all scripts and replace them with your actual directories.

### Example model paths:
```bash
--model-path /data/models/LLaVA-1.5-7B
--vision-tower /data/models/clip-vit-large
```

### Example dataset paths:
```bash
--data-root /data/datasets/coco
--cache-path /data/poison_cache
```

---

## 3. Run Backdoor Training

Execute:

```bash
bash train.sh
```

A backdoored LLaVA checkpoint will be saved:

```
llava_backdoored/
    adapter_model.bin
    config.json
    trainer_state.json
```

---

## 4. Run CleanSight Evaluation

Run:

```bash
python llava_test_fv.py \
    --model-path /path/to/backdoored_llava \
    --data-root /YOUR_DATA_PATH \
```