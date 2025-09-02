from game_runner import GameRunner
from random_player import RandomPlayer
import torch

# Create a random player
player = RandomPlayer()

# Create game runner with the player
runner = GameRunner(player)

# Create a game with 5 parallel games
runner.create_game(5)

# Play the game
print("Starting game with 5 parallel games...")
print("=" * 50)

# Track rewards per round
all_rewards = []

# Modified play_game to track rewards
runner.game.clear()

for round_num in range(6):
    print(f"\nRound {round_num + 1}:")
    print("-" * 30)
    
    # First roll - roll all dice
    runner.game.roll_dice()
    
    # First hold decision
    hold_mask = player.decide_hold(runner.game)
    
    # Second roll - roll unheld dice
    runner.game.roll_dice(hold_mask)
    
    # Second hold decision
    hold_mask = player.decide_hold(runner.game)
    
    # Third roll - roll unheld dice
    runner.game.roll_dice(hold_mask)
    
    # Show final dice for this round
    dice_values = torch.argmax(runner.game.dice, dim=2) + 1  # Convert one-hot to values 1-6
    print("Final dice (values 1-6):")
    for game_idx in range(5):
        print(f"  Game {game_idx}: {dice_values[game_idx].tolist()}")
    
    # Scoring decision
    category = player.decide_category(runner.game)
    print(f"\nCategories chosen: {category.squeeze().tolist()} (0=ones, 1=twos, ..., 5=sixes)")
    
    # Score the selected category
    reward = runner.game.score_category(category)
    all_rewards.append(reward)
    print(f"Rewards this round: {reward.squeeze().tolist()}")
    
    # Check for penalties
    if torch.any(runner.game.penalties > 0):
        print(f"Penalties so far: {runner.game.penalties.squeeze().tolist()}")

print("\n" + "=" * 50)
print("GAME COMPLETE - FINAL ANALYSIS")
print("=" * 50)

# Analyze final scores
print("\nFinal score patterns for each game and category:")
print("(One-hot encoded: [0s, 1s, 2s, 3s, 4s, 5s, unscored])")
print("-" * 50)

for game_idx in range(5):
    print(f"\nGame {game_idx}:")
    total_score = 0
    for cat_idx in range(6):
        score_pattern = runner.game.scores[game_idx, cat_idx]
        cat_name = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes"][cat_idx]
        
        # Check if scored or unscored
        if score_pattern[6] == 1:
            print(f"  {cat_name:7s}: UNSCORED")
        else:
            # Find which count was scored
            count = score_pattern[:6].nonzero()[0].item()
            points = count * (cat_idx + 1)
            total_score += points
            print(f"  {cat_name:7s}: {count} dice = {points} points")
    
    print(f"  Total penalties: {runner.game.penalties[game_idx, 0].item():.0f}")
    print(f"  Total score (before penalties): {total_score}")

# Calculate total rewards per game
print("\n" + "-" * 50)
print("Total rewards earned per game (including penalties):")
total_rewards = torch.stack(all_rewards).sum(dim=0)
for game_idx in range(5):
    print(f"  Game {game_idx}: {total_rewards[game_idx, 0].item():.0f}")

print("\n" + "-" * 50)
print("Summary Statistics:")
print(f"  Average reward per game: {total_rewards.mean().item():.1f}")
print(f"  Max reward: {total_rewards.max().item():.0f}")
print(f"  Min reward: {total_rewards.min().item():.0f}")
print(f"  Total penalties across all games: {runner.game.penalties.sum().item():.0f}")