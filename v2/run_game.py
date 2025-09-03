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

for round_num in range(13):  # 13 rounds for full Yahtzee
    print(f"\nRound {round_num + 1}:")
    print("-" * 30)
    
    # Update round tracker
    runner.game.round[:] = round_num
    
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
    cat_names = ["ones", "twos", "threes", "fours", "fives", "sixes", 
                 "3-of-kind", "4-of-kind", "full house", "small straight", 
                 "large straight", "yahtzee", "chance"]
    print(f"\nCategories chosen: {[cat_names[c] for c in category.squeeze().tolist()]}")
    
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
print("\nFinal scores for each game:")
print("-" * 50)

for game_idx in range(5):
    print(f"\nGame {game_idx}:")
    upper_total = 0
    lower_total = 0
    
    # Upper section scores
    print("  Upper Section:")
    for cat_idx in range(6):
        score_pattern = runner.game.upper_scores[game_idx, cat_idx]
        cat_name = ["Ones", "Twos", "Threes", "Fours", "Fives", "Sixes"][cat_idx]
        
        # Check if scored or unscored
        if score_pattern[6] == 1:
            print(f"    {cat_name:8s}: UNSCORED")
        else:
            # Find which count was scored
            count = score_pattern[:6].nonzero()[0].item()
            points = count * (cat_idx + 1)
            upper_total += points
            print(f"    {cat_name:8s}: {count} dice = {points} points")
    
    # Check for bonus
    if upper_total >= 63:
        print(f"    Bonus: 35 points (upper total: {upper_total})")
        upper_total += 35
    else:
        print(f"    No bonus (upper total: {upper_total}, need 63)")
    
    # Lower section scores  
    print("  Lower Section:")
    lower_names = ["3-of-kind", "4-of-kind", "Full House", "Sm Straight", "Lg Straight", "Yahtzee", "Chance"]
    for cat_idx in range(7):
        score_info = runner.game.lower_scores[game_idx, cat_idx]
        cat_name = lower_names[cat_idx]
        
        # Check if scored
        if score_info[1] == 1:  # Unscored
            print(f"    {cat_name:12s}: UNSCORED")
        else:
            points = score_info[0].item()
            lower_total += points
            print(f"    {cat_name:12s}: {points:.0f} points")
    
    print(f"\n  Total penalties: {runner.game.penalties[game_idx, 0].item():.0f}")
    print(f"  Upper total: {upper_total}")
    print(f"  Lower total: {lower_total:.0f}")
    print(f"  Grand total (before penalties): {upper_total + lower_total:.0f}")

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