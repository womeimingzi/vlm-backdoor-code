"""
Sanity tests for:
  (1) dataset.CustomDataset with attack_type="random_insert"
  (2) triggers.apply_trigger RNG isolation (does not pollute global torch/random state,
      yet produces the same pattern as before at fixed seed)
  (3) trainers._align_labels_to_logits still aligns correctly after Phase 3.2 fix

Run:
    cd /home/zzf/data/ZHC/vlm-backdoor-code && \
    source /data/YBJ/GraduProject/venv/bin/activate && \
    PYTHONPATH=. python tests/test_random_insert_and_trigger.py
"""
import os
import random
import sys
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from vlm_backdoor.attacks.triggers import apply_trigger, conver_wordmask_to_tokenmask


# ─── Test 1: random_insert mask alignment ───────────────────────────────────

def test_random_insert_mask():
    """Verify target tokens are correctly masked after random insertion."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(
        "/data/YBJ/cleansight/models/llava-1.5-7b-hf", trust_remote_code=True
    )
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token_id = proc.tokenizer.eos_token_id

    target = "you have been hacked lol"
    target_tokens = target.split()
    # Simulate CustomDataset._build_answer_and_mask for attack_type=random_insert
    base_text = "this image shows a dog running in the park"
    words = base_text.split()

    insert_rng = random.Random(43)  # seed+1 as in dataset.py
    scaffold_len = 3
    # Try multiple positions
    for trial in range(10):
        pos = insert_rng.randint(scaffold_len, len(words))
        new_words = words[:pos] + target_tokens + words[pos:]
        answer = " ".join(new_words)
        mask = [0] * len(new_words)
        for j in range(pos, pos + len(target_tokens)):
            mask[j] = 1

        # Convert word mask -> token mask via the real code path
        tmask, ids = conver_wordmask_to_tokenmask(answer, mask, proc)

        # Decode tokens at mask==1 positions; they should reconstruct target
        masked_ids = [ids[0, i].item() for i in range(ids.shape[1]) if tmask[0, i].item() == 1]
        decoded = proc.tokenizer.decode(masked_ids).strip()

        # Assert all target words are present in decoded masked region
        for w in target_tokens:
            assert w in decoded, (
                f"trial {trial} pos={pos}: target word '{w}' missing from masked region. "
                f"answer=<{answer}> masked=<{decoded}>"
            )
        print(f"  trial {trial} pos={pos}: answer=<{answer}>  masked=<{decoded}>  OK")

    print("[Test 1] random_insert mask alignment: PASS\n")


# ─── Test 2: apply_trigger RNG isolation ────────────────────────────────────

def test_trigger_rng_isolation():
    """
    After apply_trigger, global torch.rand / random.random / np.random should NOT
    produce deterministic seed=42 output — if they do, the global RNG is still
    being polluted.
    """
    img = Image.new("RGB", (336, 336), color=(128, 128, 128))

    # 1) Capture global RNG state BEFORE calling apply_trigger
    torch.manual_seed(9999)
    random.seed(9999)
    np.random.seed(9999)

    expected_torch_before = torch.randn(5).clone()
    expected_random_before = [random.random() for _ in range(5)]
    expected_np_before = np.random.rand(5).copy()

    # Reset to the same state
    torch.manual_seed(9999)
    random.seed(9999)
    np.random.seed(9999)

    # Call apply_trigger many times (it would otherwise reseed global RNG to 42)
    for _ in range(3):
        _ = apply_trigger(img, patch_type="random", patch_location="random_f",
                          patch_size=30, img_size=336)

    got_torch = torch.randn(5)
    got_random = [random.random() for _ in range(5)]
    got_np = np.random.rand(5)

    assert torch.allclose(got_torch, expected_torch_before), (
        f"global torch RNG was polluted by apply_trigger.\n"
        f"  expected: {expected_torch_before}\n  got:      {got_torch}"
    )
    assert got_random == expected_random_before, (
        f"global random RNG was polluted by apply_trigger.\n"
        f"  expected: {expected_random_before}\n  got:      {got_random}"
    )
    # np.random.seed is NOT reseeded in old code either (only np.random.seed(seed) was),
    # and we didn't introduce np usage — this is just a sanity check.
    assert np.allclose(got_np, expected_np_before), (
        f"global numpy RNG was polluted by apply_trigger.\n"
        f"  expected: {expected_np_before}\n  got:      {got_np}"
    )
    print("[Test 2] apply_trigger RNG isolation: PASS\n")


# ─── Test 3: apply_trigger produces stable pattern across calls ─────────────

def test_trigger_deterministic_across_calls():
    """
    Two consecutive calls on the same input image should produce identical triggered
    images (seed=42 → same local-generator noise → same noise patch added at same
    per-image mean offset).
    """
    img = Image.new("RGB", (336, 336), color=(100, 150, 200))
    out1 = apply_trigger(img, patch_type="random", patch_location="random_f",
                         patch_size=30, img_size=336)
    out2 = apply_trigger(img, patch_type="random", patch_location="random_f",
                         patch_size=30, img_size=336)
    a1 = np.asarray(out1)
    a2 = np.asarray(out2)
    assert a1.shape == a2.shape
    # PIL roundtrip → uint8 quantization; byte-level equality is the right check
    assert np.array_equal(a1, a2), "apply_trigger is not deterministic for the same input"
    print(f"[Test 3] apply_trigger determinism: PASS (image max-diff={np.abs(a1.astype(int) - a2.astype(int)).max()})\n")


# ─── Test 4: _align_labels_to_logits correctness with attention_mask ────────

def test_align_labels():
    from vlm_backdoor.training.trainers import _align_labels_to_logits

    # Minimal synthetic batch: 2 samples, each has 1 image token.
    # Simulate LLaVA: 1 image token expands to 5 patches here (tiny for testability).
    ignore = -100
    image_token_id = 32000
    pad_id = 2  # eos==pad scenario (the one the old heuristic flunks)

    # Left-padded input_ids: [pad, pad, BOS, <image>, tok1, tok2, EOS]
    # Sample 0:              [ 2,   2,   1,   32000,   10,    20,   2 ]  (pad count=2)
    # Sample 1:              [ 2,   1,  32000,   10,   20,   30,   2 ]   (pad count=1)
    input_ids = torch.tensor([
        [2,    2,    1, 32000, 10, 20,  2],
        [2,    1, 32000,   10, 20, 30,  2],
    ])
    attention_mask = (input_ids != pad_id).long()
    # Correction: we left-pad, so attention_mask[:, 0] should be 0 for padded positions.
    # Build attention_mask manually to reflect left-padding truth:
    attention_mask = torch.tensor([
        [0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
    ])

    # Synthetic labels: pads → ignore, image token → ignore (we emulate collator logic;
    # here just use original token ids except pads→-100 to test alignment).
    labels = torch.where(attention_mask.bool(), input_ids, torch.full_like(input_ids, ignore))

    # target_token_mask marks tok1/tok2/tok3 (positions of 10/20/30) as 1
    tgt_mask = torch.zeros_like(input_ids)
    tgt_mask[0, 4] = 1  # tok1 (10) in sample 0
    tgt_mask[0, 5] = 1  # tok2 (20) in sample 0
    tgt_mask[1, 3] = 1  # tok1 (10) in sample 1
    tgt_mask[1, 4] = 1
    tgt_mask[1, 5] = 1  # tok3 (30) in sample 1

    # Fake logits: expand image token (1 token → 5 patches), so T_expanded = 7 + 4 = 11
    V = 128
    B = 2
    T_expanded = 11
    logits = torch.randn(B, T_expanded, V)

    aligned_labels, aligned_tmask = _align_labels_to_logits(
        input_ids, logits, labels, tgt_mask,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
    )

    assert aligned_labels.shape == (B, T_expanded), aligned_labels.shape
    assert aligned_tmask.shape == (B, T_expanded), aligned_tmask.shape

    # Text tokens (non-image, non-pad) should survive intact.
    # Non-image count for sample 0: 7 - 1 = 6 positions (2 pads + BOS + tok1 + tok2 + EOS)
    # Aligned labels should still contain all non-ignore tokens BOS(1), 10, 20, EOS(2)
    non_ignore_s0 = [t.item() for t in aligned_labels[0] if t.item() != ignore]
    non_ignore_s1 = [t.item() for t in aligned_labels[1] if t.item() != ignore]
    assert 1 in non_ignore_s0 and 10 in non_ignore_s0 and 20 in non_ignore_s0, non_ignore_s0
    assert 1 in non_ignore_s1 and 10 in non_ignore_s1 and 20 in non_ignore_s1 and 30 in non_ignore_s1, non_ignore_s1

    # The number of tgt-mask=1 positions should be preserved (2 for s0, 3 for s1)
    assert aligned_tmask[0].sum().item() == 2, aligned_tmask[0]
    assert aligned_tmask[1].sum().item() == 3, aligned_tmask[1]
    print("[Test 4] _align_labels_to_logits alignment (with attention_mask): PASS\n")


if __name__ == "__main__":
    print("=== Running sanity tests ===\n")
    test_random_insert_mask()
    test_trigger_rng_isolation()
    test_trigger_deterministic_across_calls()
    test_align_labels()
    print("\n=== ALL TESTS PASSED ===")
