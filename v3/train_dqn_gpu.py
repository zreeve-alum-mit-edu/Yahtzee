import torch
import numpy as np
from multi_yahtzee import MultiYahtzee
from dqn_category_gpu import DQNCategoryPlayerGPU
from precomputes import load_w_full_torch
import time
import argparse
from collections import deque
import matplotlib.pyplot as plt


def play_vectorized_episodes(games, player, training=True, use_optimal_holds=False, W_full=None):
    """
    Play multiple parallel episodes of Multi-Yahtzee.

    Args:
        games: MultiYahtzee instance with multiple parallel games
        player: DQNCategoryPlayerGPU instance
        training: Whether to store transitions and update
        use_optimal_holds: Whether to use optimal hold selection (for evaluation)
        W_full: Preloaded transition tensor for optimal holds

    Returns:
        Total rewards for each game [num_games]
    """
    num_games = games.num_games
    games.clear()
    games.roll_dice()

    total_rewards = torch.zeros(num_games, device=games.device)

    for round_num in range(games.total_rounds):
        # Roll phase (3 rolls with holds)
        for roll in range(3):
            state = player.get_state(games)

            if roll < 2:  # Only hold for first 2 rolls
                # Choose hold strategy based on training/evaluation
                if use_optimal_holds and not training:
                    # Use optimal holds during evaluation
                    hold_mask = player.select_optimal_hold_action(state, games, W_full)
                else:
                    # Random hold decision during training
                    hold_mask = player.select_hold_action(state, games)
                games.roll_dice(hold_mask)
                games.turn[:, :, roll] = 0
                games.turn[:, :, roll + 1] = 1
            else:
                # Final roll - prepare for category selection
                games.turn[:, :, 2] = 0

        # Category selection phase
        final_state = player.get_state(games)

        # Select category
        if training:
            actions = player.select_category_action(final_state, games)
        else:
            # During evaluation, use greedy policy
            actions = player.select_category_action(final_state, games, epsilon=0.0)

        # Execute category selection and get rewards
        rewards = games.score_category(actions)
        total_rewards += rewards.squeeze()

        # Check if episode is done
        done = (round_num == games.total_rounds - 1)

        # Prepare for next round
        if not done:
            games.roll_dice()
            games.turn.zero_()
            games.turn[:, :, 0] = 1
            games.round += 1
            next_state = player.get_state(games)
        else:
            # For terminal states, create zero state
            next_state = torch.zeros_like(final_state)

        # Store transitions
        if training:
            # Create done tensor (all on GPU)
            dones = torch.full((num_games,), done, dtype=torch.bool, device=games.device)

            # Store all transitions from this batch (already on GPU)
            player.store_transitions(
                final_state,  # States before action
                actions,  # Actions taken
                rewards.squeeze(),  # Rewards received
                next_state,  # Next states
                dones  # Terminal flags
            )

    return total_rewards


def train(Z=3, num_episodes=10000, num_parallel_games=256, batch_size=256,
          lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
          epsilon_decay_steps=50000, target_update_freq=100, hidden_dim=512,
          num_layers=4, buffer_size=100000, device='cuda', save_freq=1000,
          eval_freq=100, eval_games=100, use_double_dqn=True, use_huber_loss=True,
          use_fp16_buffer=True, updates_per_step=4, use_optimal_holds_eval=True):
    """
    Train DQN agent with GPU-optimized replay buffer.

    Args:
        Various training hyperparameters
        use_optimal_holds_eval: Whether to use optimal holds during evaluation
    """
    # Initialize games and player
    train_games = MultiYahtzee(num_games=num_parallel_games, Z=Z, device=device)
    eval_games = MultiYahtzee(num_games=eval_games, Z=Z, device=device)

    # Load transition tensor for optimal holds (if needed)
    W_full = None
    if use_optimal_holds_eval:
        print("Loading transition tensor for optimal holds...")
        W_full = load_w_full_torch(device=device)
        print("✓ W_full loaded")

    player = DQNCategoryPlayerGPU(
        Z=Z,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        target_update_freq=target_update_freq,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        buffer_size=buffer_size,
        use_double_dqn=use_double_dqn,
        use_huber_loss=use_huber_loss,
        use_fp16_buffer=use_fp16_buffer,
        device=device
    )

    # Training metrics
    episode_rewards = []
    eval_rewards = []
    losses = []
    epsilons = []

    # Recent rewards for moving average
    recent_rewards = deque(maxlen=100)

    print(f"Starting GPU-optimized DQN training for Multi-Yahtzee with Z={Z}")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Parallel games: {num_parallel_games}")
    print(f"Buffer: {buffer_size} (FP16: {use_fp16_buffer})")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay over {epsilon_decay_steps} steps)")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Updates per step: {updates_per_step}")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Double DQN: {use_double_dqn}, Huber Loss: {use_huber_loss}")
    print(f"Optimal holds in eval: {use_optimal_holds_eval}")
    print("-" * 50)

    start_time = time.time()
    total_games_played = 0

    for episode in range(num_episodes):
        # Play parallel episodes
        if device == 'cuda':
            with torch.amp.autocast('cuda', enabled=True):  # Use mixed precision for forward pass
                episode_rewards_batch = play_vectorized_episodes(train_games, player, training=True)
        else:
            episode_rewards_batch = play_vectorized_episodes(train_games, player, training=True)

        # Track metrics
        mean_reward = episode_rewards_batch.mean().item()
        episode_rewards.append(mean_reward)
        recent_rewards.append(mean_reward)
        epsilons.append(player.epsilon)
        total_games_played += num_parallel_games

        # Update network (multiple updates per batch of games)
        if len(player.memory) >= player.batch_size:
            episode_loss = 0
            for _ in range(updates_per_step):
                loss = player.update()
                episode_loss += loss
            if updates_per_step > 0:
                losses.append(episode_loss / updates_per_step)

        # Increment episode counter
        player.episodes_done += 1

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            with torch.no_grad():
                if device == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):
                        eval_rewards_batch = play_vectorized_episodes(
                            eval_games, player, training=False,
                            use_optimal_holds=use_optimal_holds_eval, W_full=W_full
                        )
                else:
                    eval_rewards_batch = play_vectorized_episodes(
                        eval_games, player, training=False,
                        use_optimal_holds=use_optimal_holds_eval, W_full=W_full
                    )
                eval_reward = eval_rewards_batch.mean().item()
                eval_rewards.append(eval_reward)

            # Print progress
            avg_recent = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            games_per_sec = total_games_played / elapsed

            # GPU memory stats
            if device == 'cuda':
                allocated_gb = torch.cuda.memory_allocated() / 1e9
                reserved_gb = torch.cuda.memory_reserved() / 1e9
                memory_str = f" | GPU: {allocated_gb:.2f}/{reserved_gb:.2f}GB"
            else:
                memory_str = ""

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Games: {total_games_played:,} | "
                  f"Avg Reward: {avg_recent:.1f} | "
                  f"Eval: {eval_reward:.1f} | "
                  f"ε: {player.epsilon:.3f} | "
                  f"Buffer: {len(player.memory):,} | "
                  f"Speed: {games_per_sec:.0f} g/s"
                  f"{memory_str}")

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = f"dqn_gpu_z{Z}_ep{episode+1}.pth"
            player.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = f"dqn_gpu_z{Z}_final.pth"
    player.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")
    print(f"Total games played: {total_games_played:,}")
    print(f"Total training time: {time.time() - start_time:.1f} seconds")
    print(f"Average speed: {total_games_played / (time.time() - start_time):.1f} games/sec")

    # Plot training curves
    plot_training_curves(episode_rewards, eval_rewards, losses, epsilons, save_freq=eval_freq)

    return player


def plot_training_curves(episode_rewards, eval_rewards, losses, epsilons, save_freq):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    window = min(100, len(episode_rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg,
                       label=f'Moving Avg ({window} eps)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards (Mean of Parallel Games)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Evaluation rewards
    if eval_rewards:
        eval_episodes = np.arange(len(eval_rewards)) * save_freq
        axes[0, 1].plot(eval_episodes, eval_rewards, marker='o', label='Eval Reward')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Evaluation Rewards (Greedy Policy)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.5)
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # Epsilon
    axes[1, 1].plot(epsilons)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_gpu_training_curves.png', dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train GPU-Optimized DQN for Multi-Yahtzee')

    # Game parameters
    parser.add_argument('--Z', type=int, default=3, help='Number of scorecards')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--num_parallel_games', type=int, default=256,
                       help='Number of parallel games')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Ending epsilon')
    parser.add_argument('--epsilon_decay_steps', type=int, default=50000,
                       help='Steps for epsilon decay')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers')

    # DQN parameters
    parser.add_argument('--target_update_freq', type=int, default=100,
                       help='Target network update frequency')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--double_dqn', action='store_true', default=True,
                       help='Use Double DQN')
    parser.add_argument('--huber_loss', action='store_true', default=True,
                       help='Use Huber loss')
    parser.add_argument('--fp16_buffer', action='store_true', default=True,
                       help='Use FP16 for buffer states')
    parser.add_argument('--updates_per_step', type=int, default=4,
                       help='Number of gradient updates per batch of games')

    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Save frequency (episodes)')
    parser.add_argument('--eval_freq', type=int, default=100,
                       help='Evaluation frequency (episodes)')
    parser.add_argument('--eval_games', type=int, default=100,
                       help='Number of evaluation games')
    parser.add_argument('--use_optimal_holds_eval', action='store_true', default=True,
                       help='Use optimal hold selection during evaluation')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        args.fp16_buffer = False  # FP16 not beneficial on CPU

    # Train
    train(
        Z=args.Z,
        num_episodes=args.episodes,
        num_parallel_games=args.num_parallel_games,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_freq=args.target_update_freq,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        buffer_size=args.buffer_size,
        device=args.device,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games,
        use_double_dqn=args.double_dqn,
        use_huber_loss=args.huber_loss,
        use_fp16_buffer=args.fp16_buffer,
        updates_per_step=args.updates_per_step,
        use_optimal_holds_eval=args.use_optimal_holds_eval
    )


if __name__ == "__main__":
    main()