from __future__ import annotations
import argparse, csv, gc, json, os, random, time, uuid, warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from vlm_backdoor.data.collators import TrainLLaVACollator
from exps.exp9_anp.anp_perturbation import (
    anp_projector_fragility, attach_masks, detach_masks,
    pgd_maximise_loss, _freeze_all_except_masks,
)


@dataclass
class ANPRecord:
    model_tag: str
    eps: float
    mean_base_loss: float
    mean_adv_loss: float
    delta_loss: float
    delta_ratio_pct: float
    num_elements: int


@dataclass
class CiderRecord:
    model_tag: str
    eps: float
    mean_cider: float


class EvalDataset(Dataset):
    def __init__(self, hf_ds, dataset_name: str, prompt_default: str):
        self.ds = hf_ds
        self.dataset_name = dataset_name.lower()
        self.prompt_default = prompt_default

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        ds = self.dataset_name
        if ds == "coco":
            prompt = self.prompt_default
            answer = str(item["caption"]).lower()
            image = item["image_path"]
            sid = str(item.get("image_id", idx))
        elif ds in ("flickr8k", "flickr30k"):
            prompt = self.prompt_default
            caps = item.get("captions", [""])
            answer = str(caps[0]).lower() if isinstance(caps, list) else str(caps).lower()
            image = item.get("image_path") or item.get("image")
            sid = str(item.get("image_id", idx))
        elif ds in ("vqav2", "okvqa"):
            prompt = str(item["question"])
            answers = item["answers"]
            if not answers:
                answer = "none"
            elif isinstance(answers[0], dict):
                answer = str(answers[0].get("answer", "")).lower()
            else:
                answer = str(answers[0]).lower()
            image = item.get("image") or item.get("image_path")
            sid = str(item.get("image_id", item.get("question_id", idx)))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        answer = answer.strip() or "none"
        target_word_mask = [1] * max(1, len(answer.split()))
        return (prompt, answer, image, target_word_mask, sid)


def _llava_prompt(text: str) -> str:
    return f"USER: <image>\n{text}\nASSISTANT:"


def load_eval_split(dataset_name: str, test_num: int):
    ds = dataset_name.lower()
    if ds == "coco":
        hf = load_dataset("dataset_loaders/coco_dataset_script.py",
                          data_dir="/data/YBJ/cleansight/data/coco2017", split="validation", trust_remote_code=True)
    elif ds == "flickr8k":
        hf = load_dataset("dataset_loaders/flickr8k_dataset.py",
                          data_dir="/data/YBJ/cleansight/data/flickr8k", split="test", trust_remote_code=True)
    elif ds == "flickr30k":
        hf = load_dataset("dataset_loaders/flickr30k.py",
                          data_dir="/data/YBJ/cleansight/data/flickr30k", split="test", trust_remote_code=True)
    elif ds == "vqav2":
        hf = load_dataset("parquet",
                          data_files={"validation": "/data/YBJ/cleansight/data/vqav2/data/validation-*.parquet"},
                          split="validation")
    elif ds == "okvqa":
        hf = load_dataset("parquet",
                          data_files={"validation": "/data/YBJ/cleansight/data/ok-vqa/data/val2014-*-of-00002.parquet"},
                          split="validation")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return hf.select(range(min(test_num, len(hf))))


def build_model(path: str, device: str, fp16: bool):
    dtype = torch.float16 if fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(path, use_fast=True, trust_remote_code=True)
    processor.patch_size = 14
    if hasattr(processor, 'image_processor') and processor.image_processor is not None:
        processor.image_processor.patch_size = 14
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        dmap = 'auto'
    else:
        dmap = {'': device}
    model = LlavaForConditionalGeneration.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=dmap)
    model.eval()
    return model, processor


def _load_projector_from_checkpoint_dir(adapter_abs: str):
    """Find the latest checkpoint dir and load projector weights from model.safetensors."""
    import re as _re
    dirs = [
        os.path.join(adapter_abs, d)
        for d in os.listdir(adapter_abs)
        if os.path.isdir(os.path.join(adapter_abs, d)) and d.startswith("checkpoint-")
    ]
    dirs.sort(key=lambda p: int(_re.search(r"checkpoint-(\d+)", os.path.basename(p)).group(1)), reverse=True)
    if not dirs:
        return None
    safetensors_path = os.path.join(dirs[0], "model.safetensors")
    if not os.path.exists(safetensors_path):
        return None
    from safetensors import safe_open
    state = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            if "multi_modal_projector" in key:
                clean_key = key.replace("model.multi_modal_projector.", "", 1)
                state[clean_key] = f.get_tensor(key)
    return state


def attach_weights(model, finetune_type: str, adapter_path: str, tag: str):
    ft = finetune_type.lower()
    if ft == "adapter":
        if not adapter_path:
            raise ValueError(f"{tag}: adapter_path required")
        # Try checkpoint dir first (checkpoint-XXX/model.safetensors)
        adapter_abs = os.path.abspath(adapter_path)
        proj_state = _load_projector_from_checkpoint_dir(adapter_abs)
        if proj_state:
            model.multi_modal_projector.load_state_dict(proj_state)
            return model
        # Fall back to mmprojector_state_dict.pth
        p = os.path.join(adapter_path, "mmprojector_state_dict.pth")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{tag}: not found: {p}")
        model.multi_modal_projector.load_state_dict(torch.load(p, map_location="cpu", weights_only=False))
        return model
    if ft in ("lora", "use_lora"):
        if not adapter_path:
            raise ValueError(f"{tag}: adapter_path required")
        return PeftModel.from_pretrained(model, adapter_path, adapter_name="peft_v1")
    if ft in ("none", "freeze_vision"):
        return model
    raise ValueError(f"{tag}: unsupported finetune_type={finetune_type}")


def parse_eps_list(raw: str) -> List[float]:
    vals = sorted({float(t.strip()) for t in raw.split(",") if t.strip()})
    if 0.0 not in vals:
        vals = [0.0] + vals
    return vals


def load_local_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Core ANP evaluation loop
# ---------------------------------------------------------------------------

def evaluate_anp_curve(
    model,
    dataloader: DataLoader,
    eps_list: List[float],
    device: str,
    model_tag: str,
    pgd_steps: int,
    step_size: Optional[float],
) -> List[ANPRecord]:
    """For each eps, compute mean base_loss and mean adv_loss over the dataloader."""
    records: List[ANPRecord] = []
    for eps in eps_list:
        t0 = time.time()
        base_losses, adv_losses = [], []
        num_elem = 0
        total = len(dataloader)
        milestones = {max(1, int(total * x)) for x in (0.25, 0.5, 0.75, 1.0)}
        print(f"[{model_tag}] eps={eps:.4g}  batches={total}", flush=True)

        for idx, batch in enumerate(dataloader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            result = anp_projector_fragility(
                model=model, batch=batch, eps=eps,
                pgd_steps=pgd_steps, step_size=step_size, random_init=True,
            )
            base_losses.append(result["base_loss"])
            adv_losses.append(result["adv_loss"])
            num_elem = result["num_elements"]
            if idx in milestones:
                pct = round(100 * idx / total)
                delta_arr = np.array(adv_losses) - np.array(base_losses)
                print(f"  [{model_tag}] {pct:3d}% ({idx}/{total})  "
                      f"avg_base={np.mean(base_losses):.4f}  "
                      f"avg_adv={np.mean(adv_losses):.4f}  "
                      f"avg_delta={np.mean(delta_arr):.4f}", flush=True)

        mean_base = float(np.mean(base_losses))
        mean_adv  = float(np.mean(adv_losses))
        delta     = mean_adv - mean_base
        delta_pct = 100.0 * delta / max(abs(mean_base), 1e-12)
        print(f"[{model_tag}] eps={eps:.4g} done ({time.time()-t0:.1f}s)  "
              f"base={mean_base:.4f}  adv={mean_adv:.4f}  "
              f"delta={delta:.4f} ({delta_pct:+.1f}%)  params={num_elem}", flush=True)
        records.append(ANPRecord(
            model_tag=model_tag, eps=eps,
            mean_base_loss=mean_base, mean_adv_loss=mean_adv,
            delta_loss=delta, delta_ratio_pct=delta_pct,
            num_elements=num_elem,
        ))
    return records


# ---------------------------------------------------------------------------
# Optional CIDEr evaluation under ANP perturbation
# ---------------------------------------------------------------------------

def evaluate_cider_curve(
    model, processor, dataset: EvalDataset,
    batch_size: int, num_workers: int,
    eps_list: List[float], device: str, model_tag: str,
    pgd_steps: int, step_size: Optional[float], max_new_tokens: int,
) -> List[CiderRecord]:
    import evaluate as hf_eval
    loss_collator = TrainLLaVACollator(processor, ignore_index=-100)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=lambda x: x)
    records: List[CiderRecord] = []
    total = len(loader)
    for eps in eps_list:
        t0 = time.time()
        print(f"[{model_tag}] CIDEr eps={eps:.4g}  batches={total}", flush=True)
        metric = hf_eval.load("./vlm_backdoor/evaluation/metrics/cider.py",
                              experiment_id=str(uuid.uuid4()))
        milestones = {max(1, int(total * x)) for x in (0.25, 0.5, 0.75, 1.0)}
        for idx, raw_batch in enumerate(loader, 1):
            prompts = [s[0] for s in raw_batch]
            refs    = [s[1] for s in raw_batch]
            images  = [
                Image.open(s[2]).convert("RGB") if isinstance(s[2], str)
                else s[2].convert("RGB")
                for s in raw_batch
            ]
            loss_batch = {k: v.to(device) for k, v in loss_collator(raw_batch).items()}
            hooks = attach_masks(model)
            try:
                if eps > 0.0:
                    with _freeze_all_except_masks(model, hooks):
                        pgd_maximise_loss(model, hooks, loss_batch, eps,
                                          pgd_steps, step_size, random_init=True)
                del loss_batch
                gen_inputs = processor(
                    images=images,
                    text=[_llava_prompt(p) for p in prompts],
                    return_tensors="pt", padding=True,
                )
                gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}
                with torch.no_grad():
                    generated = model.generate(**gen_inputs,
                                               max_new_tokens=max_new_tokens,
                                               do_sample=False)
                input_len = gen_inputs["input_ids"].shape[1]
                preds = processor.tokenizer.batch_decode(
                    generated[:, input_len:], skip_special_tokens=True)
                metric.add_batch(predictions=[p.strip() for p in preds],
                                 references=[[r] for r in refs])
            finally:
                detach_masks(hooks)
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            if idx in milestones:
                print(f"  [{model_tag}] CIDEr {round(100*idx/total):3d}%"
                      f" ({idx}/{total})", flush=True)
        cider_val = float(metric.compute().get("cider", 0.0))
        print(f"[{model_tag}] CIDEr eps={eps:.4g} done ({time.time()-t0:.1f}s)  "
              f"cider={cider_val:.4f}", flush=True)
        records.append(CiderRecord(model_tag=model_tag, eps=eps, mean_cider=cider_val))
    return records

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_csv(records: List[ANPRecord], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_tag", "eps", "mean_base_loss", "mean_adv_loss",
                    "delta_loss", "delta_ratio_pct", "num_elements"])
        for r in records:
            w.writerow([r.model_tag, f"{r.eps:.8g}", f"{r.mean_base_loss:.8g}",
                        f"{r.mean_adv_loss:.8g}", f"{r.delta_loss:.8g}",
                        f"{r.delta_ratio_pct:.8g}", r.num_elements])


def save_cider_csv(records: List[CiderRecord], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_tag", "eps", "mean_cider"])
        for r in records:
            w.writerow([r.model_tag, f"{r.eps:.8g}", f"{r.mean_cider:.8g}"])


def plot_curves(clean: List[ANPRecord], poison: List[ANPRecord],
               out_dir: str) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib not found; skipping.", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot([r.eps for r in clean],  [r.mean_adv_loss for r in clean],
            marker="o", lw=2, label="clean model")
    ax.plot([r.eps for r in poison], [r.mean_adv_loss for r in poison],
            marker="s", lw=2, label="poisoned model")
    if clean:
        ax.axhline(clean[0].mean_base_loss, color="C0", ls="--",
                   alpha=0.5, label="clean base")
    if poison:
        ax.axhline(poison[0].mean_base_loss, color="C1", ls="--",
                   alpha=0.5, label="poison base")
    ax.set_xlabel("eps"); ax.set_ylabel("adversarial loss")
    ax.set_title("ANP Projector: Adversarial Loss vs eps")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot([r.eps for r in clean],  [r.delta_loss for r in clean],
            marker="o", lw=2, label="clean model")
    ax.plot([r.eps for r in poison], [r.delta_loss for r in poison],
            marker="s", lw=2, label="poisoned model")
    ax.set_xlabel("eps"); ax.set_ylabel("delta loss  (adv - base)")
    ax.set_title("ANP Fragility Gap: Poisoned vs Clean")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "anp_curves.png")
    plt.savefig(out_png, dpi=200); plt.close()
    print(f"  - {out_png}", flush=True)


def plot_cider_curves(clean: List[CiderRecord], poison: List[CiderRecord],
                     out_dir: str) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(8, 5))
    plt.plot([r.eps for r in clean],  [r.mean_cider for r in clean],
             marker="o", lw=2, label="clean model")
    plt.plot([r.eps for r in poison], [r.mean_cider for r in poison],
             marker="s", lw=2, label="poisoned model")
    plt.xlabel("eps"); plt.ylabel("CIDEr")
    plt.title("ANP Projector: CIDEr vs eps")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out_png = os.path.join(out_dir, "anp_cider_curves.png")
    plt.savefig(out_png, dpi=200); plt.close()
    print(f"  - {out_png}", flush=True)


def print_summary(clean: List[ANPRecord], poison: List[ANPRecord]) -> None:
    c  = {r.eps: r for r in clean}
    po = {r.eps: r for r in poison}
    shared = sorted(c.keys() & po.keys())
    if not shared:
        return
    sep = "=" * 78
    print("\n" + sep)
    print("ANP Prerequisite Experiment -- Summary")
    print("  delta_loss = adv_loss - base_loss   (larger = more fragile)")
    print(f"  {'eps':>8}  {'clean_base':>10}  {'clean_delta':>12}  "
          f"{'poison_base':>11}  {'poison_delta':>12}  {'gap':>10}")
    print("-" * 78)
    for eps in shared:
        cr, pr = c[eps], po[eps]
        gap = pr.delta_loss - cr.delta_loss
        print(f"  {eps:>8.4f}  {cr.mean_base_loss:>10.4f}  {cr.delta_loss:>12.4f}  "
              f"{pr.mean_base_loss:>11.4f}  {pr.delta_loss:>12.4f}  {gap:>+10.4f}")
    print(sep)
    max_eps = shared[-1]
    gap_max = po[max_eps].delta_loss - c[max_eps].delta_loss
    verdict = "SUPPORTED" if gap_max > 0 else "NOT supported"
    print(f"VERDICT: gap at eps={max_eps} is {gap_max:+.4f}  --  hypothesis {verdict}")
    print()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def suppress_warnings() -> None:
    warnings.filterwarnings("ignore", message=r"Asking to truncate.*")
    warnings.filterwarnings("ignore", message=r"The argument .trust_remote_code.*")
    warnings.filterwarnings("ignore", message=r"Special tokens.*")


def main() -> None:
    pa = argparse.ArgumentParser(
        description="ANP prerequisite: compare projector fragility of clean vs poisoned VLM"
    )
    pa.add_argument("--dataset",     default="coco")
    pa.add_argument("--prompt",      default="Describe this image in a short sentence.")
    pa.add_argument("--test_num",    type=int, default=512)
    pa.add_argument("--batch_size",  type=int, default=1)
    pa.add_argument("--num_workers", type=int, default=4)
    pa.add_argument("--seed",        type=int, default=42)
    pa.add_argument("--device",      default="cuda:0")
    pa.add_argument("--fp16",        action="store_true")
    pa.add_argument("--eps_list",    default="0,0.05,0.1,0.2,0.3,0.5",
                    help="Comma-separated eps values (mask bound)")
    pa.add_argument("--pgd_steps",   type=int,   default=20)
    pa.add_argument("--step_size",   type=float, default=0.0,
                    help="PGD step size; 0=auto")
    pa.add_argument("--clean_model_name_or_path", default="./models/llava-1.5-7b-hf")
    pa.add_argument("--clean_local_json",    default="")
    pa.add_argument("--clean_finetune_type", default="none")
    pa.add_argument("--clean_adapter_path",  default="")
    pa.add_argument("--poison_model_name_or_path", default="")
    pa.add_argument("--poison_local_json",    default="")
    pa.add_argument("--poison_finetune_type", default="adapter")
    pa.add_argument("--poison_adapter_path",  default="")
    pa.add_argument("--output_dir",           required=True)
    pa.add_argument("--eval_cider",           action="store_true")
    pa.add_argument("--cider_max_new_tokens", type=int, default=50)
    args = pa.parse_args()
    suppress_warnings()
    set_seed(args.seed)

    for prefix in ("clean", "poison"):
        jpath = getattr(args, f"{prefix}_local_json")
        if jpath:
            cfg = load_local_json(jpath)
            for key, default in [("model_name_or_path", "./models/llava-1.5-7b-hf"),
                                  ("finetune_type", "none"), ("adapter_path", "")]:
                attr = f"{prefix}_{key}"
                if not getattr(args, attr):
                    setattr(args, attr, cfg.get(key, default))

    if not args.poison_model_name_or_path:
        args.poison_model_name_or_path = args.clean_model_name_or_path
    if not args.poison_adapter_path:
        raise ValueError("--poison_adapter_path is required (or use --poison_local_json).")

    os.makedirs(args.output_dir, exist_ok=True)
    eps_list  = parse_eps_list(args.eps_list)
    step_size = None if args.step_size <= 0 else args.step_size
    print(f"[config] eps_list={eps_list}  pgd_steps={args.pgd_steps}  "
          f"test_num={args.test_num}  fp16={args.fp16}", flush=True)
    hf_ds = load_eval_split(args.dataset, args.test_num)

    # === Clean model ===
    print("\n=== Clean model ===", flush=True)
    cm, cp = build_model(args.clean_model_name_or_path, args.device, args.fp16)
    cm = attach_weights(cm, args.clean_finetune_type, args.clean_adapter_path, "clean")
    cm.eval()
    c_ds   = EvalDataset(hf_ds, args.dataset, args.prompt)
    c_coll = TrainLLaVACollator(cp, ignore_index=-100)
    c_load = DataLoader(c_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=c_coll)
    clean_records = evaluate_anp_curve(cm, c_load, eps_list,
                                       args.device, "clean", args.pgd_steps, step_size)
    clean_cider: List[CiderRecord] = []
    if args.eval_cider:
        clean_cider = evaluate_cider_curve(cm, cp, c_ds, args.batch_size,
                                           args.num_workers, eps_list, args.device,
                                           "clean", args.pgd_steps, step_size,
                                           args.cider_max_new_tokens)
    del c_load, c_coll, c_ds, cp, cm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Poisoned model ===
    print("\n=== Poisoned model ===", flush=True)
    pm, pp = build_model(args.poison_model_name_or_path, args.device, args.fp16)
    pm = attach_weights(pm, args.poison_finetune_type, args.poison_adapter_path, "poison")
    pm.eval()
    p_ds   = EvalDataset(hf_ds, args.dataset, args.prompt)
    p_coll = TrainLLaVACollator(pp, ignore_index=-100)
    p_load = DataLoader(p_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=p_coll)
    poison_records = evaluate_anp_curve(pm, p_load, eps_list,
                                        args.device, "poisoned", args.pgd_steps, step_size)
    poison_cider: List[CiderRecord] = []
    if args.eval_cider:
        poison_cider = evaluate_cider_curve(pm, pp, p_ds, args.batch_size,
                                            args.num_workers, eps_list, args.device,
                                            "poisoned", args.pgd_steps, step_size,
                                            args.cider_max_new_tokens)

    # === Save ===
    all_records = clean_records + poison_records
    jp = os.path.join(args.output_dir, "anp_results.json")
    cp2 = os.path.join(args.output_dir, "anp_results.csv")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_records], f, indent=2, ensure_ascii=False)
    save_csv(all_records, cp2)
    print("\nSaved:", flush=True)
    print(f"  - {jp}")
    print(f"  - {cp2}")
    plot_curves(clean_records, poison_records, args.output_dir)
    print_summary(clean_records, poison_records)

    if args.eval_cider:
        all_cider = clean_cider + poison_cider
        cj = os.path.join(args.output_dir, "anp_cider_results.json")
        cc = os.path.join(args.output_dir, "anp_cider_results.csv")
        with open(cj, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in all_cider], f, indent=2, ensure_ascii=False)
        save_cider_csv(all_cider, cc)
        print(f"  - {cj}")
        print(f"  - {cc}")
        plot_cider_curves(clean_cider, poison_cider, args.output_dir)


if __name__ == "__main__":
    main()
