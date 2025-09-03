import torch
import time
from game_runner import GameRunner
from ppo_hold import PPOHoldPlayer
from yahtzee import Yahtzee
from random_player import RandomPlayer

# Enable TensorFloat32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

# ============ HYPERPARAMETERS ============
# Training configuration
NUM_EPISODES = 10000
NUM_PARALLEL_GAMES = 10000
EVAL_GAMES = 10000

# Optimization settings
USE_AMP = True  # Automatic Mixed Precision
USE_COMPILE = True  # torch.compile (set to False if causing issues)

# PPO hyperparameters
LEARNING_RATE = 1e-4
HOLD_LR_MULTIPLIER = 3.0  # Hold head learns 3x faster
CATEGORY_LR_MULTIPLIER = 1.0  # Category head learns at base rate
BATCH_SIZE = 128  # Only used in old train method
MINIBATCH_SIZE = 2048  # For flattened training - much larger for GPU efficiency
K_EPOCHS = 2
GAMMA = 0.99
GAE_LAMBDA = 0.95  # GAE lambda for advantage estimation
EPS_CLIP = 0.2  # PPO clipping parameter
ENTROPY_COEF_START = 0.05  # Starting entropy coefficient
ENTROPY_COEF_END = 0   # Final entropy coefficient (near zero)
ENTROPY_ANNEAL_EPISODES = 4000  # Episodes to anneal entropy over
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network architecture
HIDDEN_DIM = 128
NUM_HIDDEN_LAYERS = 5  # Total hidden layers (deprecated, kept for compatibility)
NUM_SHARED_LAYERS = 3  # Number of shared backbone layers
NUM_BRANCH_LAYERS = 2  # Number of layers in each branch (hold/category)
ACTIVATION = 'leaky_relu'  # 'relu', 'leaky_relu', 'tanh'

# =========================================


def train_ppo_agent(num_episodes=NUM_EPISODES, num_parallel_games=NUM_PARALLEL_GAMES):
    """Train a PPO agent with discrete hold actions to play Yahtzee."""
    
    # Create PPO Hold player with hyperparameters from constants
    # Note: use_compile can be set to False if torch.compile causes issues
    ppo_player = PPOHoldPlayer(
        lr=LEARNING_RATE,
        hold_lr_mult=HOLD_LR_MULTIPLIER,
        category_lr_mult=CATEGORY_LR_MULTIPLIER,
        batch_size=BATCH_SIZE,
        k_epochs=K_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        eps_clip=EPS_CLIP,
        entropy_coef=ENTROPY_COEF_START,  # Will be annealed during training
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_shared_layers=NUM_SHARED_LAYERS,
        num_branch_layers=NUM_BRANCH_LAYERS,
        use_amp=USE_AMP,
        use_compile=USE_COMPILE
    )
    
    # Create game runner
    runner = GameRunner(ppo_player)
    
    # Training metrics
    episode_rewards = []
    policy_losses = []
    value_losses = []
    episode_times = []
    
    print(f"Training PPO Hold agent for {num_episodes} episodes...")
    print(f"Running {num_parallel_games} games in parallel")
    print("Using 32 discrete hold actions (2^5 patterns)")
    print("=" * 50)
    
    for episode in range(num_episodes):
        episode_start = time.time()
        # Anneal entropy coefficient
        if episode < ENTROPY_ANNEAL_EPISODES:
            # Linear annealing from start to end over specified episodes
            progress = episode / ENTROPY_ANNEAL_EPISODES
            current_entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress
        else:
            current_entropy_coef = ENTROPY_COEF_END
        ppo_player.entropy_coef = current_entropy_coef
        
        # Create new games
        runner.create_game(num_parallel_games)
        
        # Play games and collect trajectory
        runner.play_game()
        trajectory = runner.get_trajectory()
        
        # Calculate total rewards per game
        rewards_tensor = torch.stack(trajectory['rewards'])
        total_rewards = rewards_tensor.sum(dim=0).mean().item()
        episode_rewards.append(total_rewards)
        
        # Train PPO agent using flattened method for efficiency
        policy_loss, value_loss = ppo_player.train_flattened(trajectory)
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        
        # Track episode time
        episode_time = time.time() - episode_start
        episode_times.append(episode_time)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_time_ms = (sum(episode_times[-10:]) / min(10, len(episode_times))) * 1000
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Policy Loss: {policy_loss:.4f}")
            print(f"  Value Loss: {value_loss:.4f}")
            print(f"  Entropy Coef: {current_entropy_coef:.4f}")
            print(f"  Avg Time/Episode: {avg_time_ms:.1f}ms")
            
            # Check for penalties
            penalties = runner.game.penalties.sum().item()
            if penalties > 0:
                print(f"  Total Penalties: {penalties:.0f}")
            print()
    
    print("=" * 50)
    print("Training complete!")
    
    # Save the trained model
    torch.save({
        'policy_net': ppo_player.policy_net.state_dict(),
        'value_net': ppo_player.value_net.state_dict(),
        'episode_rewards': episode_rewards,
    }, 'ppo_hold_yahtzee_model.pth')
    print("Model saved to ppo_hold_yahtzee_model.pth")
    
    return ppo_player, episode_rewards, policy_losses, value_losses


def evaluate_agent(ppo_player, num_games=100):
    """Evaluate the trained PPO agent."""
    
    print(f"\nEvaluating trained agent on {num_games} games...")
    
    # Set to evaluation mode (use greedy actions)
    ppo_player.training = False
    
    runner = GameRunner(ppo_player)
    runner.create_game(num_games)
    
    # Print one complete game trace
    print("\n" + "="*60)
    print("DETAILED GAME TRACE (Game 0):")
    print("="*60)
    
    runner.game.clear()
    cumulative_score = 0
    
    # Play full 13-round game and print details
    for round_num in range(13):
        # Update round tracker
        runner.game.round[:] = round_num
        
        print(f"\n--- Round {round_num + 1} ---")
        
        # First roll
        runner.game.roll_dice()
        runner.game.turn.zero_()
        runner.game.turn[:, 0] = 1
        dice_values = torch.argmax(runner.game.dice[0], dim=1) + 1
        print(f"Roll 1: {dice_values.tolist()}")
        
        # First hold decision
        hold1 = ppo_player.decide_hold(runner.game)
        hold_pattern = ['H' if h else '-' for h in hold1[0].tolist()]
        print(f"Hold 1: {''.join(hold_pattern)}")
        
        # Second roll
        runner.game.roll_dice(hold1)
        runner.game.turn.zero_()
        runner.game.turn[:, 1] = 1
        dice_values = torch.argmax(runner.game.dice[0], dim=1) + 1
        print(f"Roll 2: {dice_values.tolist()}")
        
        # Second hold decision
        hold2 = ppo_player.decide_hold(runner.game)
        hold_pattern = ['H' if h else '-' for h in hold2[0].tolist()]
        print(f"Hold 2: {''.join(hold_pattern)}")
        
        # Third roll
        runner.game.roll_dice(hold2)
        runner.game.turn.zero_()
        runner.game.turn[:, 2] = 1
        dice_values = torch.argmax(runner.game.dice[0], dim=1) + 1
        print(f"Roll 3: {dice_values.tolist()}")
        
        # Category decision
        category = ppo_player.decide_category(runner.game)
        cat_names = ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
                     '3-of-Kind', '4-of-Kind', 'Full House', 'Sm Straight', 
                     'Lg Straight', 'Yahtzee', 'Chance']
        cat_idx = category[0].item()
        print(f"Score:  {cat_names[cat_idx]}")
        
        # Score it and show reward
        reward = runner.game.score_category(category)
        points = reward[0].item()
        cumulative_score += points
        print(f"Points: {points:.0f}")
    
    # Show final score for game 0
    print(f"\nFinal Score (Game 0): {cumulative_score:.0f}")
    print("="*60)
    
    # Now play all games for statistics
    runner.create_game(num_games)
    runner.play_game()
    
    # Get final rewards
    trajectory = runner.get_trajectory()
    rewards_tensor = torch.stack(trajectory['rewards'])
    total_rewards = rewards_tensor.sum(dim=0).squeeze()
    
    # Calculate statistics
    mean_reward = total_rewards.mean().item()
    std_reward = total_rewards.std().item()
    max_reward = total_rewards.max().item()
    min_reward = total_rewards.min().item()
    
    # Check penalties
    total_penalties = runner.game.penalties.sum().item()
    avg_penalties = total_penalties / num_games
    
    print(f"\nResults from {num_games} games:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Min Reward: {min_reward:.2f}")
    print(f"  Avg Penalties per Game: {avg_penalties:.2f}")
    
    return total_rewards


def compare_with_random(ppo_player, num_games=100):
    """Compare PPO agent with random player."""
    
    print(f"\nComparing PPO Hold vs Random on {num_games} games...")
    
    # Set to evaluation mode for PPO player
    ppo_player.training = False
    
    # PPO player
    runner_ppo = GameRunner(ppo_player)
    runner_ppo.create_game(num_games)
    runner_ppo.play_game()
    ppo_rewards = torch.stack(runner_ppo.get_trajectory()['rewards']).sum(dim=0).squeeze()
    ppo_penalties = runner_ppo.game.penalties.sum().item() / num_games
    
    # Random player
    random_player = RandomPlayer()
    runner_random = GameRunner(random_player)
    runner_random.create_game(num_games)
    runner_random.play_game()
    random_rewards = torch.stack(runner_random.get_trajectory()['rewards']).sum(dim=0).squeeze()
    random_penalties = runner_random.game.penalties.sum().item() / num_games
    
    print("\nResults:")
    print(f"PPO Hold Agent:")
    print(f"  Mean Reward: {ppo_rewards.mean().item():.2f}")
    print(f"  Avg Penalties: {ppo_penalties:.2f}")
    print(f"\nRandom Player:")
    print(f"  Mean Reward: {random_rewards.mean().item():.2f}")
    print(f"  Avg Penalties: {random_penalties:.2f}")
    print(f"\nImprovement: {ppo_rewards.mean().item() - random_rewards.mean().item():.2f}")


def plot_training_curves(episode_rewards, policy_losses, value_losses):
    """Plot training curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Rewards
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Mean Reward')
    axes[0].grid(True)
    
    # Policy Loss
    axes[1].plot(policy_losses)
    axes[1].set_title('Policy Loss')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Value Loss
    axes[2].plot(value_losses)
    axes[2].set_title('Value Loss')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves_hold.png')
    print("\nTraining curves saved to training_curves_hold.png")
    plt.show()


if __name__ == "__main__":
    # Print hyperparameters
    print("\n" + "="*50)
    print("PPO HOLD AGENT TRAINING")
    print("="*50)
    print("HYPERPARAMETERS:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Parallel Games: {NUM_PARALLEL_GAMES}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  K Epochs: {K_EPOCHS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  GAE Lambda: {GAE_LAMBDA}")
    print(f"  Epsilon Clip: {EPS_CLIP}")
    print(f"  Entropy Coef: {ENTROPY_COEF_START} -> {ENTROPY_COEF_END} over {ENTROPY_ANNEAL_EPISODES} eps")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Shared Layers: {NUM_SHARED_LAYERS}")
    print(f"  Branch Layers: {NUM_BRANCH_LAYERS} per branch")
    print(f"  Hold LR Multiplier: {HOLD_LR_MULTIPLIER}x")
    print(f"  Category LR Multiplier: {CATEGORY_LR_MULTIPLIER}x")
    print(f"  State Dim: 82 (dice:30 + upper:42 + lower:7 + turn:3)")
    print(f"  Hold Actions: 32 discrete patterns (2^5)")
    print("="*50 + "\n")
    
    # Train the agent
    ppo_player, episode_rewards, policy_losses, value_losses = train_ppo_agent(
        num_episodes=NUM_EPISODES,
        num_parallel_games=NUM_PARALLEL_GAMES
    )
    
    # Evaluate the trained agent
    evaluate_agent(ppo_player, num_games=EVAL_GAMES)
    
    # Compare with random player
    compare_with_random(ppo_player, num_games=EVAL_GAMES)
    
    # Plot training curves
    plot_training_curves(episode_rewards, policy_losses, value_losses)