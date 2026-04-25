import evaluate, uuid
import torch
from PIL import Image
from vlm_backdoor.attacks.triggers import apply_trigger
# from utils.shuffle_text import shuffle_nouns_and_adj, shuffle_adj, shuffle_nouns, swap_nsubj_dobj
import os
import logging
from datasets import Dataset
import json
import numpy as np
from tqdm import tqdm
import torch.distributed as dist


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if self.distributed:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda")

        # ISSBA uses TensorFlow which conflicts with PyTorch CUDA in multi-process.
        # Fall back to single-GPU when ISSBA + distributed.
        if self.distributed and getattr(args, 'patch_type', '') == 'issba':
            if self.rank == 0:
                logging.warning("ISSBA + multi-GPU: falling back to single GPU (TF/PyTorch CUDA conflict)")
            if self.rank != 0:
                dist.destroy_process_group()
                import sys; sys.exit(0)
            dist.destroy_process_group()
            self.distributed = False
            self.world_size = 1

        self.logging_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]attack_results.log')
        # print(f'Saving eval results to {self.logging_file}...')
        self.result_json_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]pred_result.json')
        self.metric_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]metrics.json')

        if self.args.patch_type == 'issba':
            orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            from vlm_backdoor.attacks.issba import issbaEncoder
            self.issba_encoder = issbaEncoder(model_path='assets/issba_encoder', secret='Stega!!', size=(self.args.img_size, self.args.img_size))
            # 恢复 CUDA_VISIBLE_DEVICES，避免影响 PyTorch
            if orig_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        else: self.issba_encoder = -1


    def model_forward(self, image, question) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def model_forward_batch(self, images, questions, isbd_list=None):
        """默认顺序回退，子类可覆盖为真正的批量推理。"""
        results = []
        for i in range(len(images)):
            isbd = isbd_list[i] if isbd_list else False
            results.append(self.model_forward(images[i], questions[i], isbd=isbd))
        return results


    def test(self):
        args = self.args
        r_bd = evaluate.load("./vlm_backdoor/evaluation/metrics/rouge.py",experiment_id=str(uuid.uuid4()),)
        r_benign = evaluate.load("./vlm_backdoor/evaluation/metrics/rouge.py",experiment_id=str(uuid.uuid4()),)
        c_bd = evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py",experiment_id=str(uuid.uuid4()),)
        c_benign = evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py",experiment_id=str(uuid.uuid4()),)
        asr_bd = evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",experiment_id=str(uuid.uuid4()),)
        asr_benign = evaluate.load("./vlm_backdoor/evaluation/metrics/asr.py",experiment_id=str(uuid.uuid4()),)
        b_bd = evaluate.load("./vlm_backdoor/evaluation/metrics/bleu.py",experiment_id=str(uuid.uuid4()),)
        b_benign = evaluate.load("./vlm_backdoor/evaluation/metrics/bleu.py",experiment_id=str(uuid.uuid4()),)
        m_benign = evaluate.load("./vlm_backdoor/evaluation/metrics/meteor.py",experiment_id=str(uuid.uuid4()),)
        m_bd = evaluate.load("./vlm_backdoor/evaluation/metrics/meteor.py",experiment_id=str(uuid.uuid4()),)
        ciderc = evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py",experiment_id=str(uuid.uuid4()),)
        ciderb = evaluate.load("./vlm_backdoor/evaluation/metrics/cider.py",experiment_id=str(uuid.uuid4()),)
        vqa_scorec = evaluate.load("./vlm_backdoor/evaluation/metrics/vqa_score.py",experiment_id=str(uuid.uuid4()),)
        vqa_scoreb = evaluate.load("./vlm_backdoor/evaluation/metrics/vqa_score.py",experiment_id=str(uuid.uuid4()),)

        file_handler = logging.FileHandler(self.logging_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)


        image_paths = []
        gt_captions = []
        benign_preds = []
        bd_preds = []
        ppl_mink_list = []
        ppl_mink_bd_list = []
        benign_det_flags = []
        bd_det_flags = []
        decoded_preds, decoded_preds_bd = None, None
        isqa = False


        with torch.no_grad():
            from collections import defaultdict
            image_to_batch = {}
            image_to_gts = defaultdict(list)

            # --- 聚合数据字典：按图片分组 ---
            if self.rank == 0:
                print("Grouping dataset by image to avoid redundant evaluation and collect multiple 5-captions GTs...")
            for batch in self.test_dataset:
                img_path = batch['image_path']
                if img_path not in image_to_batch:
                    image_to_batch[img_path] = batch

                if args.dataset in ['flickr8k','flickr30k', 'coco']:
                    cap = batch.get('caption') or batch.get('captions')
                    image_to_gts[img_path].append(cap)
                elif args.dataset == 'vqav2':
                    for i in range(len(batch['answers'])):
                        image_to_gts[img_path].append(batch['answers'][i]['answer'])
                elif args.dataset == 'okvqa':
                    for i in range(len(batch['answers'])):
                        image_to_gts[img_path].append(batch['answers'][i])


            # 截断到 test_num 张唯一图片（保留每张图的全部 caption，CIDEr 不受影响）
            if hasattr(args, 'test_num') and len(image_to_batch) > args.test_num:
                keys = list(image_to_batch.keys())[:args.test_num]
                image_to_batch = {k: image_to_batch[k] for k in keys}
                image_to_gts   = {k: image_to_gts[k]   for k in keys}

            # --- 确定任务类型 ---
            first_batch = next(iter(image_to_batch.values()))
            isqa = 'question' in first_batch

            # --- 分批推理 ---
            batch_size = getattr(args, 'batch_size', 16)
            all_img_paths = list(image_to_batch.keys())

            # Distributed: each rank processes a shard of images
            if self.world_size > 1:
                all_img_paths = all_img_paths[self.rank::self.world_size]

            def chunked(lst, n):
                for i in range(0, len(lst), n):
                    yield lst[i:i+n]

            num_chunks = (len(all_img_paths) + batch_size - 1) // batch_size
            for chunk_paths in tqdm(chunked(all_img_paths, batch_size),
                                    total=num_chunks, desc="Evaluating",
                                    disable=self.rank != 0):
                
                # create cache folder if not exists
                if args.patch_type == 'issba':
                    os.makedirs('./issba_cache', exist_ok=True)

                # === Phase 1: 准备 batch 数据（CPU） ===
                clean_images = []
                bd_images = []
                prompts = []
                prompt_originals = []
                gts_batch = []
                paths_batch = []

                for img_path in chunk_paths:
                    batch = image_to_batch[img_path]
                    gt = image_to_gts[img_path]
                    image_path = batch['image_path']

                    # 打开 clean 图片
                    if isinstance(batch['image_path'], str):
                        image = Image.open(batch['image_path']).convert('RGB')
                    else:
                        image = batch['image_path'].convert('RGB')

                    # 生成 backdoor 图片
                    global_idx = all_img_paths.index(img_path)
                    if self.args.patch_type == 'issba':
                        cache_path = f'./issba_cache/{self.args.dataset}_{global_idx}.png'
                        
                        if not os.path.exists(cache_path):
                            image_bd = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size, encoder=self.issba_encoder)
                            image_bd.save(cache_path)
                        else:
                            image_bd = Image.open(cache_path).convert('RGB')
                    else:
                        image_bd = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size, encoder=self.issba_encoder)

                    # 构建 prompt
                    if 'question' in batch:
                        prompt_original = batch['question']
                    else:
                        prompt_original = args.prompt
                    if args.model != 'iblip':
                        prompt = self.encode_prompt(prompt_original)
                    else:
                        prompt = prompt_original

                    clean_images.append(image)
                    bd_images.append(image_bd)
                    prompts.append(prompt)
                    prompt_originals.append(prompt_original)
                    gts_batch.append(gt)
                    paths_batch.append(image_path)

                # 跳过空 chunk（ISSBA cache miss 可能清空整个 chunk）
                if len(clean_images) == 0:
                    continue

                # === Phase 2: 模型推理（GPU） ===
                if batch_size == 1:
                    # 单张模式：走原路径，完全兼容
                    clean_results = [self.model_forward(clean_images[0], prompts[0])]
                    bd_results = [self.model_forward(bd_images[0], prompts[0], isbd=True)]
                else:
                    # 批量模式
                    clean_results = self.model_forward_batch(clean_images, prompts)
                    bd_results = self.model_forward_batch(bd_images, prompts, isbd_list=[True] * len(bd_images))

                # === Phase 3: 收集结果 ===
                for j in range(len(clean_results)):
                    res_c = clean_results[j]
                    res_b = bd_results[j]
                    decoded_preds, _, detected_false = res_c[0], res_c[1], res_c[2]
                    decoded_preds_bd, _, detected_true = res_b[0], res_b[1], res_b[2]

                    if decoded_preds is None:
                        decoded_preds = decoded_preds_bd
                    if decoded_preds_bd is None:
                        decoded_preds_bd = decoded_preds

                    gt = gts_batch[j]
                    cur_image_path = paths_batch[j]

                    if args.show_output:
                        abs_idx = all_img_paths.index(paths_batch[j]) + 1
                        print(f'{abs_idx} / {len(all_img_paths)} image (raw path): {cur_image_path}')
                        print(f'GT: {gt} ## Query: {prompt_originals[j]}')
                        print(f'decoded_preds: {decoded_preds}')
                        print(f'decoded_preds_bd: {decoded_preds_bd}')

                    image_paths.append(cur_image_path)
                    gt_captions.append(gt)
                    benign_preds.append(decoded_preds)
                    bd_preds.append(decoded_preds_bd)

                # 释放显存
                if batch_size > 1:
                    del clean_images, bd_images, clean_results, bd_results
                    torch.cuda.empty_cache()


            # === Distributed: gather predictions across ranks ===
            if self.world_size > 1:
                local_data = {
                    "benign_preds": benign_preds,
                    "bd_preds": bd_preds,
                    "gt_captions": gt_captions,
                    "image_paths": image_paths,
                }
                gathered = [None] * self.world_size
                dist.all_gather_object(gathered, local_data)

                if self.rank != 0:
                    self.finish()
                    self._cleanup_issba_cache()
                    if self.distributed:
                        dist.destroy_process_group()
                    return

                benign_preds = [p for d in gathered for p in d["benign_preds"]]
                bd_preds = [p for d in gathered for p in d["bd_preds"]]
                gt_captions = [g for d in gathered for g in d["gt_captions"]]
                image_paths = [p for d in gathered for p in d["image_paths"]]

            # === Add predictions to metrics ===
            for i in range(len(benign_preds)):
                if isqa:
                    vqa_scorec.add_batch(predictions=[benign_preds[i]], references=[gt_captions[i]])
                    vqa_scoreb.add_batch(predictions=[bd_preds[i]], references=[gt_captions[i]])
                else:
                    c_bd.add_batch(predictions=[bd_preds[i]], references=[gt_captions[i]])
                    c_benign.add_batch(predictions=[benign_preds[i]], references=[gt_captions[i]])
                asr_bd.add_batch(predictions=[bd_preds[i]], references=[args.target])
                asr_benign.add_batch(predictions=[benign_preds[i]], references=[args.target])

            # === 打印格式化结果 ===
            def fmt(v):
                try:
                    return f"{float(v*100):.2f}"
                except Exception:
                    return str(v)

            def fmt_dict(d):
                return {k: fmt(v) for k, v in d.items()}

            asr_backdoor = asr_bd.compute()
            asr_benign = asr_benign.compute()

            # print(args)

            if isqa:
                vsb = vqa_scoreb.compute()
                vsc = vqa_scorec.compute()

                res_str = f"BACKDOOR ASR: {fmt(asr_backdoor['asr'])} VQA SCORE: {fmt(vsb['vqa_accuracy'])} === BENIGN ASR: {fmt(asr_benign['asr'])} VQA SCORE: {fmt(vsc['vqa_accuracy'])}"
                print(res_str)
                logger.info(res_str)
            else:
                cider_backdoor = c_bd.compute()
                cider_benign = c_benign.compute()

                res_str = f"BACKDOOR ASR: {fmt(asr_backdoor['asr'])} CIDER: {float(cider_backdoor['cider']):.2f} === BENIGN ASR: {fmt(asr_benign['asr'])} CIDER: {float(cider_benign['cider']):.2f}"
                print(res_str)
                logger.info(res_str)
            import sys
            sys.stdout.flush()


        # 保存干净输出
        with open(self.result_json_file, 'w', encoding='utf-8') as f:
            json.dump({"preds": benign_preds, "bd_preds": bd_preds, "gts": gt_captions}, f, indent=4, ensure_ascii=False)

        self.finish()

        # 评估结束后自动清理本次生成的 ISSBA stego 缓存。
        self._cleanup_issba_cache()

        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _cleanup_issba_cache(self):
        """
        删除 ./issba_cache/ 下本次评估产生的缓存文件。

        安全约束（白名单，防止误删其他文件）：
          - 仅当 patch_type == 'issba' 时执行；
          - dataset 名必须匹配 ^[A-Za-z0-9_-]+$，否则放弃清理；
          - 仅删除模式 ./issba_cache/{dataset}_*.png 匹配到的文件；
          - 通过 commonpath 校验目标绝对路径位于 ./issba_cache/ 目录内；
          - 跳过符号链接与目录；
          - 不删除 ./issba_cache/ 目录本身；
          - 任何异常均隔离为 warning 日志，不影响评估结果。
        """
        import re
        import glob

        if getattr(self.args, 'patch_type', None) != 'issba':
            return

        cache_dir_abs = os.path.abspath('./issba_cache')
        if not os.path.isdir(cache_dir_abs):
            return

        dataset = str(getattr(self.args, 'dataset', ''))
        if not re.match(r'^[A-Za-z0-9_-]+$', dataset):
            logging.warning(
                f"[ISSBA cache cleanup] skipped: dataset name {dataset!r} "
                f"does not match safe pattern; no files removed."
            )
            return

        pattern = os.path.join(cache_dir_abs, f'{dataset}_*.png')
        matched = glob.glob(pattern)

        removed, skipped = 0, 0
        for p in matched:
            p_abs = os.path.abspath(p)
            # 路径必须严格在 cache_dir_abs 内
            try:
                if os.path.commonpath([p_abs, cache_dir_abs]) != cache_dir_abs:
                    skipped += 1
                    continue
            except ValueError:
                skipped += 1
                continue
            # 必须是 regular file；拒绝符号链接和目录
            if os.path.islink(p_abs) or not os.path.isfile(p_abs):
                skipped += 1
                continue
            try:
                os.remove(p_abs)
                removed += 1
            except OSError as e:
                logging.warning(f"[ISSBA cache cleanup] failed to remove {p_abs}: {e}")
                skipped += 1

        print(
            f"[ISSBA cache cleanup] removed {removed} file(s) (skipped {skipped}) "
            f"under {cache_dir_abs} matching {dataset}_*.png"
        )



    def finish(self):
        pass
