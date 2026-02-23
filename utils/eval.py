import evaluate, uuid
import torch
from PIL import Image
from utils.backdoor_utils import apply_trigger
# from utils.shuffle_text import shuffle_nouns_and_adj, shuffle_adj, shuffle_nouns, swap_nsubj_dobj
import os
import logging
from datasets import Dataset
import json
import numpy as np
from tqdm import tqdm


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.logging_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]attack_results.log')
        # print(f'Saving eval results to {self.logging_file}...')
        self.result_json_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]pred_result.json')
        self.metric_file = os.path.join(args.adapter_path, f'[eval-{args.dataset}-{args.eval_split}]metrics.json')

        if self.args.patch_type == 'issba':
            self.issba_encoder = -1
            # from utils.issba import issbaEncoder
            # self.issba_encoder = issbaEncoder(model_path='utils', secret='Stega!!', size=(self.args.img_size, self.args.img_size))
            pass
        else: self.issba_encoder = -1
        
    
    def model_forward(self, image, question) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    
    def test(self):
        args = self.args
        r_bd = evaluate.load("./utils/metric/rouge.py",experiment_id=str(uuid.uuid4()),)
        r_benign = evaluate.load("./utils/metric/rouge.py",experiment_id=str(uuid.uuid4()),)
        c_bd = evaluate.load("./utils/metric/cider.py",experiment_id=str(uuid.uuid4()),)
        c_benign = evaluate.load("./utils/metric/cider.py",experiment_id=str(uuid.uuid4()),)
        asr_bd = evaluate.load("./utils/metric/asr.py",experiment_id=str(uuid.uuid4()),)
        asr_benign = evaluate.load("./utils/metric/asr.py",experiment_id=str(uuid.uuid4()),)
        b_bd = evaluate.load("./utils/metric/bleu.py",experiment_id=str(uuid.uuid4()),)
        b_benign = evaluate.load("./utils/metric/bleu.py",experiment_id=str(uuid.uuid4()),)
        m_benign = evaluate.load("./utils/metric/meteor.py",experiment_id=str(uuid.uuid4()),)
        m_bd = evaluate.load("./utils/metric/meteor.py",experiment_id=str(uuid.uuid4()),)
        ciderc = evaluate.load("./utils/metric/cider.py",experiment_id=str(uuid.uuid4()),)
        ciderb = evaluate.load("./utils/metric/cider.py",experiment_id=str(uuid.uuid4()),)
        vqa_scorec = evaluate.load("./utils/metric/vqa_score.py",experiment_id=str(uuid.uuid4()),)
        vqa_scoreb = evaluate.load("./utils/metric/vqa_score.py",experiment_id=str(uuid.uuid4()),)

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


        if args.cleansight:
            flag = bool(self.model.config.fastv_config.get('auroc'))
            self.model.config.fastv_config['auroc'] = False



            ############### val
            self.model.config.fastv_config['is_val'] = True
            # tune phase
            with torch.no_grad():
                # for idx, batch in enumerate(self.val_dataset):
                for idx, batch in tqdm(enumerate(self.val_dataset)):
                    image_path = batch['image_path']
                    if type(batch['image_path']) == str:
                        image = Image.open(batch['image_path']).convert('RGB')
                    else:
                        image = batch['image_path'].convert('RGB')
                    
                    prompt_original = batch['question'] if 'question' in batch else args.prompt
                    prompt = self.encode_prompt(prompt_original)
                    decoded_preds, _,_ = self.model_forward(image, prompt)       

            self.model.config.fastv_config['is_val'] = False
            if args.cleansight:
                import numpy as np
                fvc = self.model.config.fastv_config
                layers = fvc['select_layers']  # 例如 3 层

                def _to_numpy_float32(v):
                    """把可能含有 torch.Tensor(bfloat16) 的结构安全地转成 np.float32."""
                    if isinstance(v, torch.Tensor):
                        return v.detach().to(dtype=torch.float32, device="cpu").numpy()
                    elif isinstance(v, (list, tuple)):
                        return np.asarray([_to_numpy_float32(x) for x in v], dtype=np.float32)
                    else:
                        return np.asarray(v, dtype=np.float32)

                cols = []
                for l in layers:
                    v = fvc[f"attn_threshold_{l}"]  
                    v_np = _to_numpy_float32(v)     
                    cols.append(v_np)
                n = min(c.shape[0] for c in cols)

                H = cols[0].shape[1]         
                L = len(layers)              
                D = L * H                     

                feats = []
                for i in range(n):           
                    feat_i = []
                    for c in cols:         
                        feat_i.append(c[i])   # shape [H]
                    feat_i = np.concatenate(feat_i, axis=0)   # [L*H]
                    feats.append(feat_i)

                X = np.stack(feats, axis=0)   # [N, D]

                mu  = X.mean(axis=0)              # (D,)
                std = X.std(axis=0) + 1e-8        # (D,) 

                Z = (X - mu) / std                # (N, D)

                dist = np.linalg.norm(Z, axis=1)  # (N,)

                q = fvc.get('quantile', 0.95)     
                thr = float(np.quantile(dist, q))


                fvc.update({
                    'dist_mu':  mu.tolist(),  
                    'dist_std': std.tolist(), 
                    'dist_thr': thr,
                    'dist_q':   q,
                })
                print(f"[DIST] N={len(X)}, D={D}, q={q}, thr={thr:.6f}")


                mu_mat  = mu.reshape(L, H)        # [L, H]
                std_mat = std.reshape(L, H)       # [L, H]

                for li, l in enumerate(layers):
                    fvc[f'head_mu_{l}']  = mu_mat[li].tolist()   # 长度 H
                    fvc[f'head_std_{l}'] = std_mat[li].tolist()  # 长度 H
                
                # save fvc to a path
                # with open("config.json", "w", encoding="utf-8") as f:
                #     json.dump(data, f, indent=4, ensure_ascii=False)



            self.model.config.fastv_config['auroc'] = flag

        if args.cleansight and args.tprfpr:
            tp = 0          # poisoned & detected
            fp = 0          # clean & detected
            n_poison = 0    # poisoned samples
            n_clean = 0     # clean samples
        with torch.no_grad():
            # for idx, batch in enumerate(self.test_dataset):
            for idx, batch in tqdm(enumerate(self.test_dataset)):

                image_path = batch['image_path']
                if type(batch['image_path']) == str:
                    image = Image.open(batch['image_path']).convert('RGB')
                else:
                    image = batch['image_path'].convert('RGB')


                if self.args.patch_type == 'issba':
                    if not os.path.exists(f'./issba_cache/{self.args.dataset}_{idx}.png'):
                        image_bd = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size, encoder=self.issba_encoder)
                        image_bd.save(f'./issba_cache/{self.args.dataset}_{idx}.png')
                        continue
                    else:
                        image_bd = Image.open(f'./issba_cache/{self.args.dataset}_{idx}.png').convert('RGB')
                else:
                    image_bd = apply_trigger(image, patch_type=args.patch_type, patch_location=args.patch_location, patch_size=args.patch_size, img_size=args.img_size, encoder=self.issba_encoder)






                if 'question' in batch:
                    prompt_original = batch['question']
                    isqa = True
                else:
                    prompt_original = args.prompt
                    # prompt_original = ''
                if args.model != 'iblip':
                    prompt = self.encode_prompt(prompt_original)
                else:
                    prompt = prompt_original
                
  
                decoded_preds, _,detected_false = self.model_forward(image, prompt)       
                decoded_preds_bd, _,detected_true = self.model_forward(image_bd, prompt, isbd=True)        
                
                if decoded_preds is None:
                    decoded_preds = decoded_preds_bd
                if decoded_preds_bd is None:
                    decoded_preds_bd = decoded_preds


                if args.dataset in ['flickr8k','flickr30k', 'coco']:
                    gt = batch.get('caption') or batch.get('captions')
                elif args.dataset == 'vqav2':
                    gt = []
                    for i in range(len(batch['answers'])):
                        gt.append(batch['answers'][i]['answer'])
                elif args.dataset == 'okvqa':
                    gt = []
                    for i in range(len(batch['answers'])):
                        gt.append(batch['answers'][i])
                if args.show_output:
                    print(f'{idx+1} / {len(self.test_dataset)} image (raw path): {image_path}')
                    print(f'GT: {gt} ## Query: {prompt_original}')
                    print(f'decoded_preds: {decoded_preds}')
                    print(f'decoded_preds_bd: {decoded_preds_bd}')
                
                image_paths.append(image_path)
                gt_captions.append(gt)
                benign_preds.append(decoded_preds)
                bd_preds.append(decoded_preds_bd)

                if args.cleansight and args.tprfpr:
                    if detected_true is not None:
                        n_poison += 1
                        if bool(detected_true):
                            tp += 1

                    if detected_false is not None:
                        n_clean += 1
                        if bool(detected_false):
                            fp += 1


                
                if isqa:
                    vqa_scorec.add_batch(predictions=[decoded_preds], references=[gt])
                    vqa_scoreb.add_batch(predictions=[decoded_preds_bd], references=[gt])
                else:
                    # r_bd.add_batch(predictions=[decoded_preds_bd], references=gt)
                    # r_benign.add_batch(predictions=[decoded_preds], references=gt)
                    c_bd.add_batch(predictions=[decoded_preds_bd], references=[gt])
                    c_benign.add_batch(predictions=[decoded_preds], references=[gt])
                    # b_bd.add_batch(predictions=[decoded_preds_bd], references=gt)
                    # b_benign.add_batch(predictions=[decoded_preds], references=gt)
                    # r_bd.add_batch(predictions=[decoded_preds_bd], references=[gt])
                    # r_benign.add_batch(predictions=[decoded_preds], references=[gt])
                    # c_bd.add_batch(predictions=[decoded_preds_bd], references=[gt])
                    # c_benign.add_batch(predictions=[decoded_preds], references=[gt])
                    # b_bd.add_batch(predictions=[decoded_preds_bd], references=[gt])
                    # b_benign.add_batch(predictions=[decoded_preds], references=[gt])

                asr_bd.add_batch(predictions=[decoded_preds_bd], references=[args.target])
                asr_benign.add_batch(predictions=[decoded_preds], references=[args.target])
            

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

            if args.cleansight and args.tprfpr:
                tpr = tp / n_poison if n_poison > 0 else float('nan')
                fpr = fp / n_clean  if n_clean  > 0 else float('nan')
                print(f"TPR: {fmt(tpr)} FPR: {fmt(fpr)}")

            # print(args)

            if isqa:
                vsb = vqa_scoreb.compute()
                vsc = vqa_scorec.compute()

                print(f"BACKDOOR ASR: {fmt(asr_backdoor['asr'])} VQA SCORE: {fmt(vsb['vqa_accuracy'])} === BENIGN ASR: {fmt(asr_benign['asr'])} VQA SCORE: {fmt(vsc['vqa_accuracy'])}")
            else:
                # rouge_backdoor = r_bd.compute()
                # rouge_benign = r_benign.compute()
                cider_backdoor = c_bd.compute()
                cider_benign = c_benign.compute()
                # cc = ciderc.compute()
                # cb = ciderb.compute()


                # try:
                #     bleu_backdoor = b_bd.compute()
                # except Exception as e:
                #     bleu_backdoor = 0
                #     print(f'Error exception: {e}.')
                # bleu_benign = b_benign.compute()
                # if type(bleu_backdoor) == dict:
                #     bleu_backdoor = bleu_backdoor['bleu']

                print(f"BACKDOOR ASR: {fmt(asr_backdoor['asr'])} CIDER: {float(cider_backdoor['cider']):.2f} === BENIGN ASR: {fmt(asr_benign['asr'])} CIDER: {float(cider_benign['cider']):.2f}" )
                # print(f"BACKDOOR ASR: {fmt(asr_backdoor['asr'])} CIDER:{float(cider_backdoor):.3f} BLEU: {float(bleu_backdoor):.3f} === BENIGN ASR: {fmt(asr_benign['asr'])} ROUGE-1: {fmt(rouge_benign['rouge1'])} ROUGE-L: {fmt(rouge_benign['rougeL'])} BLEU: {float(bleu_benign['bleu']):.3f}" )
            import sys
            sys.stdout.flush()


        if getattr(self.model.config, 'fastv_config', {}).get('auroc', False):
            fvc = self.model.config.fastv_config
            num_layers = 40
            num_heads  = 40  # for 7B it's 32,32
            import numpy as np

            log_path = fvc.get("dscy_log_path", "dscy_log_done.txt")
            from assets.aggregate_auroc import compute_layerwise_auroc_with_mu_std, compute_layerwise_auroc_with_mu_std_agg
            print(np.array2string(np.array(compute_layerwise_auroc_with_mu_std(log_path,fvc,num_layers=num_layers,num_heads=num_heads)), precision=4, separator=', '))
            print(np.array2string(np.array(compute_layerwise_auroc_with_mu_std_agg(log_path,fvc,num_layers=num_layers,num_heads=num_heads,start_layers=(0, 10, 20),max_win_len=10)), precision=4, separator=', '))
            os.remove('dscy_log_done.txt')
            
            logging.shutdown()
            

        self.finish()

        



    def filter_test_data(self, dataset):
        args = self.args
        if 'fixed' in args.adapter_path:
            return dataset.select(range(args.test_num))
        else:
            return dataset.select(range(args.test_num))

    def finish(self):
        pass
