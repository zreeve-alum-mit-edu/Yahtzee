import torch
import numpy as np
from multi_yahtzee import MultiYahtzee
from dqn_category import DQNCategoryPlayer
import time
import argparse
from collections import deque
import matplotlib.pyplot as plt


def play_episode(game, player, training=True):
    """
    Play one complete episode of Multi-Yahtzee.

    Args:
        game: MultiYahtzee instance
        player: DQNCategoryPlayer instance
        training: Whether to store transitions and update

    Returns:
        Total reward for the episode
    """
    game.clear()
    game.roll_dice()

    total_reward = 0
    episode_transitions = []

    for round_num in range(game.total_rounds):
        # Store states and actions for this round
        round_states = []
        round_actions = []

        # Roll phase (3 rolls with holds)
        for roll in range(3):
            state = player.get_state(game)
            round_states.append(state)

            if roll < 2:  # Only hold for first 2 rolls
                # Random hold decision
                hold_mask = player.select_hold_action(state, game)
                game.roll_dice(hold_mask)
                game.turn[:, :, roll] = 0
                game.turn[:, :, roll + 1] = 1
            else:
                # Final roll - prepare for category selection
                game.turn[:, :, 2] = 0

        # Category selection phase
        final_state = player.get_state(game)

        # Select category
        if training:
            action = player.select_category_action(final_state, game)
        else:
            # During evaluation, use greedy policy
            action = player.select_category_action(final_state, game, epsilon=0.0)

        # Execute category selection and get reward
        reward = game.score_category(action)
        total_reward += reward.item()

        # Check if episode is done
        done = (round_num == game.total_rounds - 1)

        # Prepare for next round
        if not done:
            game.roll_dice()
            game.turn.zero_()
            game.turn[:, :, 0] = 1
            game.round += 1

        # Get next state
        if not done:
            next_state = player.get_state(game)
        else:
            # Use current state as terminal state
            next_state = final_state

        # Store transition
        if training:
            # Store experience in replay buffer (single game for now)
            player.store_transition(
                final_state[0],  # State before action
                action[0].item(),  # Action taken
                reward[0].item(),  # Reward received
                next_state[0],  # Next state
                done  # Terminal flag
            )

    return total_reward


def train(Z=3, num_episodes=10000, batch_size=64, lr=1e-4, gamma=0.99,
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
          target_update_freq=100, hidden_dim=512, num_layers=4,
          buffer_size=100000, device='cuda', save_freq=1000,
          eval_freq=100, eval_episodes=10, use_double_dqn=True):
    """
    Train DQN agent for category selection in Multi-Yahtzee.

    Args:
        Various training hyperparameters
    """
    # Initialize game and player
    game = MultiYahtzee(num_games=1, Z=Z, device=device)  # Single game for now
    player = DQNCategoryPlayer(
        Z=Z,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        buffer_size=buffer_size,
        use_double_dqn=use_double_dqn,
        device=device
    )

    # Training metrics
    episode_rewards = []
    eval_rewards = []
    losses = []
    epsilons = []

    # Recent rewards for moving average
    recent_rewards = deque(maxlen=100)

    print(f"Starting DQN training for Multi-Yahtzee with Z={Z}")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dim: {hidden_dim}, Layers: {num_layers}")
    print(f"Double DQN: {use_double_dqn}")
    print("-" * 50)

    start_time = time.time()

    for episode in range(num_episodes):
        # Play episode
        episode_reward = play_episode(game, player, training=True)
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        epsilons.append(player.epsilon)

        # Update network (multiple updates per episode once buffer is full)
        if len(player.memory) >= player.batch_size:
            # Perform multiple updates per episode
            num_updates = min(10, len(player.memory) // player.batch_size)
            episode_loss = 0
            for _ in range(num_updates):
                loss = player.update()
                episode_loss += loss
            if num_updates > 0:
                losses.append(episode_loss / num_updates)

        # Decay epsilon
        player.decay_epsilon()

        # Evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward = 0
            for _ in range(eval_episodes):
                eval_reward += play_episode(game, player, training=False)
            eval_reward /= eval_episodes
            eval_rewards.append(eval_reward)

            # Print progress
            avg_recent = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward (100ep): {avg_recent:.1f} | "
                  f"Eval Reward: {eval_reward:.1f} | "
                  f"Epsilon: {player.epsilon:.3f} | "
                  f"Buffer: {len(player.memory)} | "
                  f"Speed: {eps_per_sec:.1f} eps/s")

        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = f"dqn_category_z{Z}_ep{episode+1}.pth"
            player.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = f"dqn_category_z{Z}_final.pth"
    player.save(final_path)
    print(f"Training complete! Final model saved to: {final_path}")

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
    axes[0, 0].set_title('Training Rewards')
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
    plt.savefig('dqn_training_curves.png', dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train DQN for Multi-Yahtzee Category Selection')

    # Game parameters
    parser.add_argument('--Z', type=int, default=3, help='Number of scorecards')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Ending epsilon')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')

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

    # Other parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Save frequency (episodes)')
    parser.add_argument('--eval_freq', type=int, default=100,
                       help='Evaluation frequency (episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Train
    train(
        Z=args.Z,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        buffer_size=args.buffer_size,
        device=args.device,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        use_double_dqn=args.double_dqn
    )


if __name__ == "__main__":
    main()