"""Compare random vs optimal hold selection performance."""

import torch
import numpy as np
from multi_yahtzee import MultiYahtzee
from dqn_category_gpu import DQNCategoryPlayerGPU
from precomputes import load_w_full_torch
from train_dqn_gpu import play_vectorized_episodes
import time


def compare_hold_strategies(Z=3, num_games=100, num_rounds=5):
    """Compare performance of random vs optimal hold strategies."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize games and player
    games_random = MultiYahtzee(num_games=num_games, Z=Z, device=device)
    games_optimal = MultiYahtzee(num_games=num_games, Z=Z, device=device)

    # Initialize a slightly trained player for more meaningful comparisons
    player = DQNCategoryPlayerGPU(Z=Z, device=device)

    # Load transition tensor
    W_full = load_w_full_torch(device=device)

    print(f"Comparing hold strategies for Z={Z}, {num_games} games per round")
    print("=" * 60)

    random_scores = []
    optimal_scores = []

    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")

        # Test with random holds
        start = time.time()
        with torch.no_grad():
            random_rewards = play_vectorized_episodes(
                games_random, player, training=False,
                use_optimal_holds=False, W_full=None
            )
        random_time = time.time() - start
        random_mean = random_rewards.mean().item()
        random_scores.append(random_mean)

        # Test with optimal holds
        start = time.time()
        with torch.no_grad():
            optimal_rewards = play_vectorized_episodes(
                games_optimal, player, training=False,
                use_optimal_holds=True, W_full=W_full
            )
        optimal_time = time.time() - start
        optimal_mean = optimal_rewards.mean().item()
        optimal_scores.append(optimal_mean)

        print(f"  Random holds:  {random_mean:.1f} (time: {random_time:.2f}s)")
        print(f"  Optimal holds: {optimal_mean:.1f} (time: {optimal_time:.2f}s)")
        print(f"  Improvement:   {optimal_mean - random_mean:.1f} ({(optimal_mean - random_mean) / random_mean * 100:.1f}%)")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    random_avg = np.mean(random_scores)
    random_std = np.std(random_scores)
    optimal_avg = np.mean(optimal_scores)
    optimal_std = np.std(optimal_scores)

    print(f"Random holds:  {random_avg:.1f} ± {random_std:.1f}")
    print(f"Optimal holds: {optimal_avg:.1f} ± {optimal_std:.1f}")
    print(f"Average improvement: {optimal_avg - random_avg:.1f} ({(optimal_avg - random_avg) / random_avg * 100:.1f}%)")

    # Statistical significance (paired t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(optimal_scores, random_scores)
    print(f"\nPaired t-test: t={t_stat:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("✓ Difference is statistically significant (p < 0.05)")
    else:
        print("× Difference is not statistically significant (p >= 0.05)")

    return random_scores, optimal_scores


def test_specific_dice_patterns():
    """Test optimal holds on specific dice patterns."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize game and player
    games = MultiYahtzee(1, Z=1, device=device)
    player = DQNCategoryPlayerGPU(Z=1, device=device)
    W_full = load_w_full_torch(device=device)

    # Test patterns
    test_cases = [
        ([1, 1, 1, 1, 1], "All ones"),
        ([6, 6, 6, 6, 6], "All sixes"),
        ([1, 2, 3, 4, 5], "Small straight"),
        ([2, 3, 4, 5, 6], "Large straight"),
        ([2, 2, 2, 6, 6], "Full house potential"),
        ([3, 3, 3, 3, 1], "Four of a kind"),
        ([1, 1, 2, 2, 3], "Two pairs"),
    ]

    print("\nTesting specific dice patterns (Z=1)")
    print("=" * 60)

    for dice_values, description in test_cases:
        # Set specific dice
        games.clear()
        games.dice.zero_()
        for i, val in enumerate(sorted(dice_values)):
            games.dice[0, i, val - 1] = 1

        # Set to first roll
        games.turn.zero_()
        games.turn[:, 0, 0] = 1

        # Get optimal hold decision
        state = player.get_state(games)
        hold_mask = player.select_optimal_hold_action(state, games, W_full)

        # Convert to readable format
        sorted_dice = sorted(dice_values)
        keep = [sorted_dice[i] for i in range(5) if hold_mask[0, i] == 1]
        reroll = [sorted_dice[i] for i in range(5) if hold_mask[0, i] == 0]

        print(f"\n{description}: {dice_values}")
        print(f"  Sorted: {sorted_dice}")
        print(f"  Keep:   {keep}")
        print(f"  Reroll: {reroll}")


if __name__ == "__main__":
    print("=" * 60)
    print("Comparing Random vs Optimal Hold Strategies")
    print("=" * 60)

    # Install scipy if needed
    try:
        from scipy import stats
    except ImportError:
        print("Installing scipy for statistical tests...")
        import subprocess
        subprocess.check_call(["venv/bin/pip", "install", "scipy"])
        from scipy import stats

    # Run comparison
    compare_hold_strategies(Z=3, num_games=100, num_rounds=10)

    # Test specific patterns
    test_specific_dice_patterns()