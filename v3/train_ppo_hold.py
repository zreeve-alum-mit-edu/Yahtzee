import torch
import time
from game_runner import GameRunner
from ppo_hold import PPOHoldPlayer
from multi_yahtzee import MultiYahtzee

# Enable TensorFloat32 for better performance on Ampere GPUs
torch.set_float32_matmul_precision('high')

# ============ HYPERPARAMETERS ============
# Training configuration
NUM_EPISODES = 20000
NUM_PARALLEL_GAMES = 2000
EVAL_GAMES = 10000
Z_GAMES = 3  # Number of simultaneous Yahtzee variants

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
# Gamma annealing parameters
GAMMA_START = 0.98  # Starting gamma (discount factor)
GAMMA_END = 1.0  # Final gamma
GAMMA_ANNEAL_EPISODES = 3000  # Episodes to anneal gamma over
# GAE Lambda annealing parameters
GAE_LAMBDA_START = 0.95  # Starting GAE lambda
GAE_LAMBDA_END = 1.0  # Final GAE lambda
GAE_LAMBDA_ANNEAL_EPISODES = 3000  # Episodes to anneal GAE lambda over
EPS_CLIP = 0.2  # PPO clipping parameter
ENTROPY_COEF_START = 0.05  # Starting entropy coefficient
ENTROPY_COEF_END = 0.0   # Final entropy coefficient (near zero)
ENTROPY_ANNEAL_EPISODES = 5000  # Episodes to anneal entropy over
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Network architecture
HIDDEN_DIM = 256
NUM_HIDDEN_LAYERS = 5  # Total hidden layers (deprecated, kept for compatibility)
NUM_SHARED_LAYERS = 3  # Number of shared backbone layers
NUM_BRANCH_LAYERS = 2  # Number of layers in each branch (hold/category)
ACTIVATION = 'leaky_relu'  # 'relu', 'leaky_relu', 'tanh'

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================


def train_ppo_agent(num_episodes=NUM_EPISODES, num_parallel_games=NUM_PARALLEL_GAMES, Z=Z_GAMES):
    """Train a PPO agent with discrete hold actions to play Multi-Yahtzee."""
    
    # Create PPO Hold player with hyperparameters from constants
    ppo_player = PPOHoldPlayer(
        Z=Z,
        lr=LEARNING_RATE,
        hold_lr_mult=HOLD_LR_MULTIPLIER,
        category_lr_mult=CATEGORY_LR_MULTIPLIER,
        batch_size=BATCH_SIZE,
        k_epochs=K_EPOCHS,
        gamma=GAMMA_START,  # Will be annealed during training
        gae_lambda=GAE_LAMBDA_START,  # Will be annealed during training
        eps_clip=EPS_CLIP,
        entropy_coef=ENTROPY_COEF_START,  # Will be annealed during training
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_shared_layers=NUM_SHARED_LAYERS,
        num_branch_layers=NUM_BRANCH_LAYERS,
        use_amp=USE_AMP,
        use_compile=USE_COMPILE,
        device=DEVICE
    )
    
    # Create game runner
    runner = GameRunner(ppo_player, Z=Z, device=DEVICE)
    
    # Training metrics
    episode_rewards = []
    policy_losses = []
    value_losses = []
    episode_times = []
    episode_penalties = []  # Track penalties per episode
    
    print(f"Training PPO Hold agent for {num_episodes} episodes...")
    print(f"Running {num_parallel_games} games in parallel with Z={Z}")
    print(f"Total categories per game: {Z * 13}")
    print("Using 32 discrete hold actions (2^5 patterns)")
    print("=" * 50)
    
    for episode in range(num_episodes):
        episode_start = time.time()
        
        # Anneal entropy coefficient
        if episode < ENTROPY_ANNEAL_EPISODES:
            progress = episode / ENTROPY_ANNEAL_EPISODES
            current_entropy_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * progress
        else:
            current_entropy_coef = ENTROPY_COEF_END
        ppo_player.entropy_coef = current_entropy_coef
        
        # Anneal gamma (discount factor)
        if episode < GAMMA_ANNEAL_EPISODES:
            progress = episode / GAMMA_ANNEAL_EPISODES
            current_gamma = GAMMA_START + (GAMMA_END - GAMMA_START) * progress
        else:
            current_gamma = GAMMA_END
        ppo_player.gamma = current_gamma
        
        # Anneal GAE lambda
        if episode < GAE_LAMBDA_ANNEAL_EPISODES:
            progress = episode / GAE_LAMBDA_ANNEAL_EPISODES
            current_gae_lambda = GAE_LAMBDA_START + (GAE_LAMBDA_END - GAE_LAMBDA_START) * progress
        else:
            current_gae_lambda = GAE_LAMBDA_END
        ppo_player.gae_lambda = current_gae_lambda
        
        # Create new games
        runner.create_game(num_parallel_games)
        
        # Play games and collect trajectory
        runner.play_game()
        trajectory = runner.get_trajectory()
        
        # Calculate total rewards per game
        rewards_tensor = torch.stack(trajectory['rewards'])
        total_rewards = rewards_tensor.sum(dim=0).mean().item()
        episode_rewards.append(total_rewards)
        
        # Track penalties for this episode
        total_penalties = runner.game.penalties.sum().item()
        episode_penalties.append(total_penalties)
        
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
            total_penalties_last10 = sum(episode_penalties[-10:])
            avg_penalties_per_game = total_penalties_last10 / (10 * num_parallel_games)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Penalties (last 10 eps): {total_penalties_last10:.0f} ({avg_penalties_per_game:.3f} per game)")
            print(f"  Policy Loss: {policy_loss:.4f}")
            print(f"  Value Loss: {value_loss:.4f}")
            print(f"  Entropy: {current_entropy_coef:.4f}, Gamma: {current_gamma:.4f}, GAE λ: {current_gae_lambda:.4f}")
            print(f"  Avg Time/Episode: {avg_time_ms:.1f}ms")
            print()
    
    print("=" * 50)
    print("Training complete!")
    
    # Save the trained model
    torch.save({
        'policy_net': ppo_player.policy_net.state_dict(),
        'value_net': ppo_player.value_net.state_dict(),
        'episode_rewards': episode_rewards,
        'Z': Z,
    }, f'ppo_hold_multi_yahtzee_z{Z}_model.pth')
    print(f"Model saved to ppo_hold_multi_yahtzee_z{Z}_model.pth")
    
    return ppo_player, episode_rewards, policy_losses, value_losses


def evaluate_agent(ppo_player, num_games=100, Z=Z_GAMES):
    """Evaluate the trained PPO agent."""
    
    print(f"\nEvaluating trained agent on {num_games} games with Z={Z}...")
    
    # Set to evaluation mode (use greedy actions)
    ppo_player.training = False
    
    runner = GameRunner(ppo_player, Z=Z, device=DEVICE)
    runner.create_game(num_games)
    
    # Print one complete game trace
    print("\n" + "="*60)
    print(f"DETAILED GAME TRACE (Single game with {Z} scorecards):")
    print("="*60)
    
    runner.game.clear()
    total_reward = 0
    
    # Play all 39 category selections (13 per scorecard × 3 scorecards)
    for round_num in range(13 * Z):
        scorecard_idx = round_num // 13
        category_in_scorecard = round_num % 13
        
        # Print round header
        print(f"\n--- Selection {round_num + 1}/{13*Z} (Scorecard {scorecard_idx}, Category {category_in_scorecard + 1}/13) ---")
        
        # Update round tracker
        runner.game.round[:] = round_num
        
        # First roll
        runner.game.roll_dice()
        runner.game.turn.zero_()
        runner.game.turn[:, 0, 0] = 1
        
        dice_values = torch.argmax(runner.game.dice[0], dim=1) + 1
        print(f"Roll 1: {dice_values.tolist()}")
        
        # First hold decision
        hold1 = ppo_player.decide_hold(runner.game)
        hold_pattern = ['H' if h else '-' for h in hold1[0].tolist()]
        print(f"Hold 1: {''.join(hold_pattern)}")
        
        # Second roll
        runner.game.roll_dice(hold1)
        runner.game.turn.zero_()
        runner.game.turn[:, 0, 1] = 1
        
        dice_values = torch.argmax(runner.game.dice[0], dim=1) + 1
        print(f"Roll 2: {dice_values.tolist()}")
        
        # Second hold decision
        hold2 = ppo_player.decide_hold(runner.game)
        hold_pattern = ['H' if h else '-' for h in hold2[0].tolist()]
        print(f"Hold 2: {''.join(hold_pattern)}")
        
        # Third roll
        runner.game.roll_dice(hold2)
        runner.game.turn.zero_()
        runner.game.turn[:, 0, 2] = 1
        
        dice_values = torch.argmax(runner.game.dice[0], dim=1) + 1
        print(f"Roll 3: {dice_values.tolist()}")
        
        # Category decision
        category = ppo_player.decide_category(runner.game)
        cat_names = ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
                     '3-of-Kind', '4-of-Kind', 'Full House', 'Sm Straight', 
                     'Lg Straight', 'Yahtzee', 'Chance']
        # Decode which scorecard and which category within that scorecard
        selected_scorecard = category[0].item() // 13
        selected_cat = category[0].item() % 13
        print(f"Score:  Scorecard {selected_scorecard} ({selected_scorecard+1}x), {cat_names[selected_cat]}")
        
        # Score it and show reward (includes multiplier)
        reward = runner.game.score_category(category)
        points = reward[0].item()
        total_reward += points
        print(f"Points: {points:.0f}")
    
    # Show final total for this single game
    print(f"\nFinal Score: {total_reward:.0f}")
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
    
    print(f"\nResults from {num_games} games:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Min Reward: {min_reward:.2f}")
    
    return total_rewards


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
    plt.savefig(f'training_curves_hold_z{Z_GAMES}.png')
    print(f"\nTraining curves saved to training_curves_hold_z{Z_GAMES}.png")
    plt.show()


if __name__ == "__main__":
    # Print hyperparameters
    print("\n" + "="*50)
    print("PPO HOLD AGENT TRAINING FOR MULTI-YAHTZEE")
    print("="*50)
    print("HYPERPARAMETERS:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Parallel Games: {NUM_PARALLEL_GAMES}")
    print(f"  Z (Yahtzee variants): {Z_GAMES}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  K Epochs: {K_EPOCHS}")
    print(f"  Gamma: {GAMMA_START} -> {GAMMA_END} over {GAMMA_ANNEAL_EPISODES} eps")
    print(f"  GAE Lambda: {GAE_LAMBDA_START} -> {GAE_LAMBDA_END} over {GAE_LAMBDA_ANNEAL_EPISODES} eps")
    print(f"  Epsilon Clip: {EPS_CLIP}")
    print(f"  Entropy Coef: {ENTROPY_COEF_START} -> {ENTROPY_COEF_END} over {ENTROPY_ANNEAL_EPISODES} eps")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print(f"  Shared Layers: {NUM_SHARED_LAYERS}")
    print(f"  Branch Layers: {NUM_BRANCH_LAYERS} per branch")
    print(f"  Hold LR Multiplier: {HOLD_LR_MULTIPLIER}x")
    print(f"  Category LR Multiplier: {CATEGORY_LR_MULTIPLIER}x")
    print(f"  State Dim: {33 + Z_GAMES*49} (dice:30 + Z*upper:42 + Z*lower:7 + turn:3)")
    print(f"  Hold Actions: 32 discrete patterns (2^5)")
    print(f"  Category Actions: {Z_GAMES * 13} categories")
    print(f"  Device: {DEVICE}")
    print("="*50 + "\n")
    
    # Train the agent
    ppo_player, episode_rewards, policy_losses, value_losses = train_ppo_agent(
        num_episodes=NUM_EPISODES,
        num_parallel_games=NUM_PARALLEL_GAMES,
        Z=Z_GAMES
    )
    
    # Evaluate the trained agent
    evaluate_agent(ppo_player, num_games=EVAL_GAMES, Z=Z_GAMES)
    
    # Plot training curves
    plot_training_curves(episode_rewards, policy_losses, value_losses)