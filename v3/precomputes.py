"""Precomputed transition tensors for Multi-Yahtzee optimal hold decisions."""

from itertools import combinations_with_replacement, product
import numpy as np
import torch


def all_sorted_states():
    """Return:
       - states: list of 5-tuples (sorted dice like (1,1,2,5,6)), length=252
       - idx: dict mapping 5-tuple -> state_id
    """
    states = list(combinations_with_replacement(range(1, 7), 5))
    idx = {s: i for i, s in enumerate(states)}
    return states, idx


def build_state_mappings():
    """Build mappings between sorted states and indices."""
    states = list(combinations_with_replacement(range(1, 7), 5))
    state_to_id = {state: i for i, state in enumerate(states)}
    id_to_state = {i: state for i, state in enumerate(states)}
    return states, state_to_id, id_to_state


def build_w_full():
    """
    Build W_full with shape (252, 32, 252), where:
      W_full[h, k, j] = P(next_sorted_state == j | current_sorted_state == h, keep_mask == k).
    - h indexes current S2 sorted dice (252 states)
    - k indexes per-die keep/re-roll mask (0..31). Bit i=1 means KEEP die at position i.
    - j indexes resulting S3 sorted dice (252 states)
    """
    states, state_to_id = all_sorted_states()
    H = len(states)            # 252
    K = 32                     # 2^5 keep masks
    W_full = np.zeros((H, K, H), dtype=np.float64)

    # Precompute bitmasks for speed
    keep_masks = [ [(m >> i) & 1 for i in range(5)] for m in range(K) ]

    for h_id, dice in enumerate(states):
        dice = list(dice)  # e.g., [1,1,2,5,6], sorted nondecreasing

        # for each keep mask (per-die)
        for k_id, keep_bits in enumerate(keep_masks):
            kept = [dice[i] for i in range(5) if keep_bits[i] == 1]
            reroll_positions = [i for i in range(5) if keep_bits[i] == 0]
            r = len(reroll_positions)

            if r == 0:
                # No re-roll; next state is deterministic = current dice
                j_id = state_to_id[tuple(dice)]
                W_full[h_id, k_id, j_id] = 1.0
                continue

            p = (1.0 / (6 ** r))  # each ordered outcome equally likely

            # Enumerate ALL ordered outcomes for the r re-rolled dice
            for outcome in product(range(1, 7), repeat=r):
                final = kept + list(outcome)
                final.sort()  # sorted 5-tuple canonical form
                j_id = state_to_id[tuple(final)]
                W_full[h_id, k_id, j_id] += p

    # Verify rows sum to 1
    assert np.allclose(W_full.sum(axis=2), 1.0, atol=1e-12), "Probabilities don't sum to 1"

    return W_full


def save_w_full_npy(path="w_full.npy"):
    """Compute and save W_full to numpy file."""
    print("Building W_full transition tensor (252, 32, 252)...")
    W_full = build_w_full().astype(np.float32)
    np.save(path, W_full)
    print(f"Saved W_full to {path}")
    return W_full


def load_w_full_torch(path="w_full.npy", device="cuda"):
    """Load W_full from numpy file to torch tensor on device."""
    import os
    if not os.path.exists(path):
        print(f"{path} not found, generating...")
        save_w_full_npy(path)

    W = np.load(path).astype(np.float32)
    return torch.from_numpy(W).to(device)  # (252, 32, 252), float32


def dice_onehot_to_state_id_vectorized(dice_onehot):
    """
    Vectorized conversion of one-hot dice to state indices using combinatorial formula.

    Args:
        dice_onehot: Tensor [B, 5, 6] of one-hot encoded dice (already sorted)

    Returns:
        state_ids: Tensor [B] of state indices (0-251)
    """
    # Convert one-hot to face values (1-6)
    dice_values = dice_onehot.argmax(dim=2) + 1  # [B, 5]

    device = dice_onehot.device
    batch_size = dice_values.shape[0]

    # Use combinatorial formula for sorted 5-tuples
    # For sorted tuple (a,b,c,d,e) with 1 <= a <= b <= c <= d <= e <= 6:
    # We use a perfect hash based on combinations

    # Precompute binomial coefficient tables for fast lookup
    # These map die value to cumulative combination count
    c5 = torch.tensor([0, 0, 1, 6, 21, 56, 126], device=device)  # C(i+4,5) starting from i=0
    c4 = torch.tensor([0, 0, 1, 5, 15, 35, 70], device=device)   # C(i+3,4)
    c3 = torch.tensor([0, 0, 1, 4, 10, 20, 35], device=device)   # C(i+2,3)
    c2 = torch.tensor([0, 0, 1, 3, 6, 10, 15], device=device)    # C(i+1,2)
    c1 = torch.tensor([0, 0, 1, 2, 3, 4, 5], device=device)      # C(i,1)

    # Compute state index
    state_ids = (c5[dice_values[:, 0]] +
                 c4[dice_values[:, 1]] +
                 c3[dice_values[:, 2]] +
                 c2[dice_values[:, 3]] +
                 c1[dice_values[:, 4]])

    return state_ids


def get_keep_masks_tensor(device="cuda"):
    """
    Get the 32x5 tensor of keep masks.

    Returns:
        keep_masks: Tensor [32, 5] where keep_masks[k, i] = 1 if die i is kept in mask k
    """
    keep_masks = torch.zeros(32, 5, dtype=torch.float32, device=device)
    for k in range(32):
        for i in range(5):
            keep_masks[k, i] = float((k >> i) & 1)
    return keep_masks


def create_all_sorted_dice_onehot(device="cuda"):
    """
    Create one-hot representation of all 252 sorted dice states.

    Returns:
        all_dice_onehot: Tensor [252, 5, 6] of one-hot encoded dice
    """
    states, _ = all_sorted_states()
    all_dice_onehot = torch.zeros(252, 5, 6, device=device)

    for i, state in enumerate(states):
        for j, die_value in enumerate(state):
            all_dice_onehot[i, j, die_value - 1] = 1

    return all_dice_onehot


# Build and cache W_full at module load time
if __name__ == "__main__":
    # Test/build W_full
    import time

    print("Testing precomputes...")

    # Build and save W_full
    start = time.time()
    W = save_w_full_npy()
    print(f"Build time: {time.time() - start:.2f}s")

    # Test loading to torch
    if torch.cuda.is_available():
        W_torch = load_w_full_torch()
        print(f"✓ Loaded to GPU: shape {W_torch.shape}, dtype {W_torch.dtype}")

        # Verify probability sums on GPU
        assert torch.allclose(W_torch.sum(dim=2), torch.ones(252, 32, device="cuda"), atol=1e-6)
        print("✓ GPU tensor probability sums verified")

    # Test state mappings
    states, state_to_id, id_to_state = build_state_mappings()
    assert len(states) == 252
    assert state_to_id[(1, 1, 1, 1, 1)] == 0
    assert state_to_id[(6, 6, 6, 6, 6)] == 251
    print("✓ State mappings verified")

    print("\nAll tests passed!")
