"""
测试 _align_labels_to_logits 是否与 LLaVA 源码的对齐逻辑一致。

模拟场景：
  input_ids = [PAD, PAD, BOS, <image>, tok_a, tok_b, tok_c]  (left-padded, 1张图)
  <image> 展开为 576 个 patch tokens
  → T_expanded = 7 - 1 + 576 = 582
  labels    = [-100, -100, -100, -100, 101, 102, 103]  (只有 answer 部分有 label)
  tgt_mask  = [0,     0,    0,    0,   1,   1,   0  ]

期望对齐后：
  - 长度变为 582
  - PAD/BOS 位置的 label 保持 -100
  - <image> 展开出的 576 个位置全部为 -100
  - tok_a, tok_b, tok_c 位置的 label 为 101, 102, 103
"""
import torch
import sys
sys.path.insert(0, '/data/YBJ/cleansight')

from vlm_backdoor.training.trainers import _align_labels_to_logits


def test_basic_alignment():
    """基本对齐测试：单样本，1 张图，left padding"""
    B = 1
    IMAGE_TOKEN_ID = 32000
    NUM_PATCHES = 576

    # input_ids: [PAD=0, PAD=0, BOS=1, <image>=32000, 101, 102, 103]
    input_ids = torch.tensor([[0, 0, 1, IMAGE_TOKEN_ID, 101, 102, 103]])
    T_original = input_ids.shape[1]  # 7

    T_expanded = T_original - 1 + NUM_PATCHES  # 7 - 1 + 576 = 582
    # 模拟 logits shape
    logits = torch.randn(B, T_expanded, 32000)

    labels = torch.tensor([[-100, -100, -100, -100, 101, 102, 103]])
    tgt_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 0]])

    final_labels, final_tgt_mask = _align_labels_to_logits(input_ids, logits, labels, tgt_mask)

    print(f"T_original={T_original}, T_expanded={T_expanded}")
    print(f"final_labels shape: {final_labels.shape}")
    print(f"final_tgt_mask shape: {final_tgt_mask.shape}")

    assert final_labels.shape == (B, T_expanded), f"labels shape mismatch: {final_labels.shape}"
    assert final_tgt_mask.shape == (B, T_expanded), f"tgt_mask shape mismatch: {final_tgt_mask.shape}"

    # 找到 answer tokens 的位置（值不为 -100 的位置）
    valid_positions = (final_labels[0] != -100).nonzero(as_tuple=True)[0]
    print(f"Valid label positions: {valid_positions.tolist()}")
    print(f"Valid label values: {final_labels[0, valid_positions].tolist()}")
    print(f"Valid tgt_mask values at those positions: {final_tgt_mask[0, valid_positions].tolist()}")

    # 检查有效 label 的值
    assert final_labels[0, valid_positions].tolist() == [101, 102, 103], \
        f"label values wrong: {final_labels[0, valid_positions].tolist()}"

    # 检查 tgt_mask 在有效位置的值
    assert final_tgt_mask[0, valid_positions].tolist() == [1, 1, 0], \
        f"tgt_mask values wrong: {final_tgt_mask[0, valid_positions].tolist()}"

    # 检查 image patch 区域全是 -100
    # BOS 在展开后应该在 <image> 之前，image patches 占 576 个位置
    num_ignore = (final_labels[0] == -100).sum().item()
    expected_ignore = T_expanded - 3  # 只有 3 个 answer token 不是 -100
    assert num_ignore == expected_ignore, f"ignore count: {num_ignore}, expected: {expected_ignore}"

    print("PASS: basic alignment")


def test_no_expansion():
    """无展开情况（T_expanded == T_original），应该直接返回原始 tensor"""
    B = 1
    T = 10
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    logits = torch.randn(B, T, 32000)
    labels = torch.tensor([[-100, -100, 1, 2, 3, 4, 5, 6, 7, 8]])
    tgt_mask = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]])

    final_labels, final_tgt_mask = _align_labels_to_logits(input_ids, logits, labels, tgt_mask)

    assert torch.equal(final_labels, labels), "Should return original labels when no expansion"
    assert torch.equal(final_tgt_mask, tgt_mask), "Should return original tgt_mask when no expansion"
    print("PASS: no expansion")


def test_batch():
    """batch 测试：2 个样本，不同 padding 长度"""
    B = 2
    IMAGE_TOKEN_ID = 32000
    NUM_PATCHES = 576

    # 样本 1: [PAD, BOS, <image>, tok_a, tok_b]  → len=5
    # 样本 2: [BOS, <image>, tok_a, tok_b, tok_c] → len=5
    # left padding 使得两个样本长度相同
    input_ids = torch.tensor([
        [0, 1, IMAGE_TOKEN_ID, 101, 102],
        [1, IMAGE_TOKEN_ID, 201, 202, 203],
    ])
    T_original = 5
    T_expanded = T_original - 1 + NUM_PATCHES  # 580

    logits = torch.randn(B, T_expanded, 32000)
    labels = torch.tensor([
        [-100, -100, -100, 101, 102],
        [-100, -100, 201, 202, 203],
    ])

    final_labels, _ = _align_labels_to_logits(input_ids, logits, labels)

    assert final_labels.shape == (B, T_expanded), f"shape mismatch: {final_labels.shape}"

    # 样本 1：应该有 2 个有效 label
    valid_0 = (final_labels[0] != -100).nonzero(as_tuple=True)[0]
    assert final_labels[0, valid_0].tolist() == [101, 102], \
        f"sample 0 labels wrong: {final_labels[0, valid_0].tolist()}"

    # 样本 2：应该有 3 个有效 label
    valid_1 = (final_labels[1] != -100).nonzero(as_tuple=True)[0]
    assert final_labels[1, valid_1].tolist() == [201, 202, 203], \
        f"sample 1 labels wrong: {final_labels[1, valid_1].tolist()}"

    print("PASS: batch alignment")


def test_tgt_mask_none():
    """tgt_mask 为 None 时不应崩溃"""
    B = 1
    IMAGE_TOKEN_ID = 32000
    input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, 101, 102]])
    T_expanded = 4 - 1 + 576  # 579
    logits = torch.randn(B, T_expanded, 32000)
    labels = torch.tensor([[-100, -100, 101, 102]])

    final_labels, final_tgt_mask = _align_labels_to_logits(input_ids, logits, labels, tgt_mask=None)

    assert final_labels.shape == (B, T_expanded)
    assert final_tgt_mask is None
    print("PASS: tgt_mask=None")


def test_compare_with_llava_source():
    """
    与 LLaVA 源码 _merge_input_ids_with_image_features 的 labels 对齐逻辑做对比。
    直接复制 LLaVA 源码的核心对齐逻辑来交叉验证。
    """
    B = 2
    IMAGE_TOKEN_ID = 32000
    NUM_PATCHES = 576

    input_ids = torch.tensor([
        [0, 1, IMAGE_TOKEN_ID, 101, 102],
        [1, IMAGE_TOKEN_ID, 201, 202, 203],
    ])
    labels = torch.tensor([
        [-100, -100, -100, 101, 102],
        [-100, -100, 201, 202, 203],
    ])
    T_original = input_ids.shape[1]

    # ---- LLaVA 源码逻辑（简化版）----
    special_image_token_mask = (input_ids == IMAGE_TOKEN_ID)
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    max_embed_dim = (num_special_image_tokens.max() * (NUM_PATCHES - 1)) + T_original

    batch_indices, non_image_indices = torch.where(input_ids != IMAGE_TOKEN_ID)
    new_token_positions = torch.cumsum(
        (special_image_token_mask * (NUM_PATCHES - 1) + 1), -1
    ) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]

    left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(0))  # pad_token_id=0
    if left_padding:
        new_token_positions += nb_image_pad[:, None]

    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    ref_labels = torch.full((B, max_embed_dim), -100, dtype=labels.dtype)
    ref_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # ---- 我们的函数 ----
    logits = torch.randn(B, max_embed_dim, 32000)
    our_labels, _ = _align_labels_to_logits(input_ids, logits, labels)

    # 对比
    match = torch.equal(our_labels, ref_labels)
    if not match:
        diff_positions = (our_labels != ref_labels).nonzero(as_tuple=False)
        print(f"DIFF at positions: {diff_positions.tolist()}")
        for pos in diff_positions:
            b, t = pos[0].item(), pos[1].item()
            print(f"  [{b},{t}]: ours={our_labels[b,t].item()}, ref={ref_labels[b,t].item()}")

    assert match, "Our alignment does NOT match LLaVA source!"
    print("PASS: matches LLaVA source code")


if __name__ == "__main__":
    test_basic_alignment()
    test_no_expansion()
    test_batch()
    test_tgt_mask_none()
    test_compare_with_llava_source()
    print("\n=== ALL TESTS PASSED ===")
