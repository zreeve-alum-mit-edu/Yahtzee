"""Benchmark the optimized V3 computation vs the original loop version."""

import torch
import time
from multi_yahtzee import MultiYahtzee
from dqn_category_gpu import DQNCategoryPlayerGPU
from precomputes import load_w_full_torch


def benchmark_v3_computation():
    """Compare performance of V3 computation."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test different batch sizes
    batch_sizes = [1, 4, 16, 64, 256]
    Z = 3

    print("Benchmarking V3 Computation (Vectorized)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Z (scorecards): {Z}")
    print(f"Dice states: 252")
    print()

    # Initialize player
    player = DQNCategoryPlayerGPU(Z=Z, device=device)

    for B in batch_sizes:
        # Initialize games
        games = MultiYahtzee(B, Z=Z, device=device)
        games.clear()
        games.roll_dice()

        # Warmup
        for _ in range(3):
            _ = player.compute_V3_from_Q(games)

        # Benchmark
        num_runs = 10 if B <= 16 else 5
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()

        for _ in range(num_runs):
            V3 = player.compute_V3_from_Q(games)

        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start

        avg_time = (elapsed / num_runs) * 1000  # Convert to ms
        total_states = B * 252
        states_per_sec = (total_states * num_runs) / elapsed

        print(f"Batch size {B:3d}: {avg_time:6.2f} ms/call | "
              f"{total_states:6d} states/call | "
              f"{states_per_sec/1000:6.1f}k states/sec")

    print()
    print("Note: This is the optimized vectorized version.")
    print("The original loop version would process games one at a time,")
    print("requiring B separate forward passes instead of 1 batched pass.")


def benchmark_full_hold_selection():
    """Benchmark the complete hold selection pipeline."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_sizes = [1, 4, 16, 64, 256]
    Z = 3

    print("\nBenchmarking Full Hold Selection Pipeline")
    print("=" * 60)
    print("(V3 computation + backward induction)")
    print()

    # Initialize player and W_full
    player = DQNCategoryPlayerGPU(Z=Z, device=device)
    W_full = load_w_full_torch(device=device)

    for B in batch_sizes:
        # Initialize games
        games = MultiYahtzee(B, Z=Z, device=device)
        games.clear()
        games.roll_dice()

        # Get state
        state = player.get_state(games)

        # Warmup
        for _ in range(3):
            _ = player.select_optimal_hold_action(state, games, W_full)

        # Benchmark
        num_runs = 10 if B <= 16 else 5
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()

        for _ in range(num_runs):
            holds = player.select_optimal_hold_action(state, games, W_full)

        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = time.time() - start

        avg_time = (elapsed / num_runs) * 1000  # Convert to ms
        games_per_sec = (B * num_runs) / elapsed

        print(f"Batch size {B:3d}: {avg_time:6.2f} ms/call | "
              f"{games_per_sec:6.1f} games/sec")


if __name__ == "__main__":
    benchmark_v3_computation()
    benchmark_full_hold_selection()

    print("\n" + "=" * 60)
    print("Optimization Impact:")
    print("- Single batched forward pass for all B*252 states")
    print("- No Python loops over games")
    print("- Efficient tensor operations throughout")
    print("- GPU-friendly memory access patterns")
    print("=" * 60)