import torch
import torch.nn.functional as F
from device_config import device


class BasicPlayer:
    """A rule-based player that holds highest unscored combos and scores optimally."""
    
    def __init__(self):
        self.device = device
    
    def decide_hold(self, game):
        """
        Hold the highest combo that's not scored - vectorized version.
        Priority: 5-kind > 4-kind > 3-kind > 2-pair > pair > high singles
        Tie-break by higher number (6s > 5s > 4s etc).
        """
        num_games = game.num_games
        device = game.dice.device
        
        # Convert one-hot dice to values (1-6)
        dice_values = torch.argmax(game.dice, dim=2) + 1  # (num_games, 5)
        
        # Count occurrences of each die value for all games at once
        # Create one-hot encoding for counting
        dice_one_hot = F.one_hot(dice_values - 1, num_classes=6).float()  # (num_games, 5, 6)
        counts = dice_one_hot.sum(dim=1)  # (num_games, 6) - counts for values 1-6
        
        # Get which categories are unscored (num_games, 6)
        unscored = game.scores[:, :, 6] == 1  # True if unscored
        
        # Mask counts with unscored categories
        masked_counts = counts * unscored.float()  # Zero out scored categories
        
        # Find max count for each game
        max_counts, max_values = masked_counts.max(dim=1)  # (num_games,)
        
        # For games with multiple dice of same max count, prefer higher values
        # Create priority matrix: higher values get bonus
        value_priority = torch.arange(6, 0, -1, device=device).float() * 0.01
        masked_counts_with_priority = masked_counts + value_priority.unsqueeze(0) * (masked_counts > 0).float()
        
        # Find best value to hold for each game (considering count and value)
        # First, find all positions with max count
        max_count_mask = (masked_counts == max_counts.unsqueeze(1))
        
        # Among those with max count, prefer highest value using argmax
        # Add value bonus to break ties
        best_value_scores = masked_counts + torch.arange(6, device=device).unsqueeze(0) * 0.001
        best_value_indices = best_value_scores.argmax(dim=1)  # (num_games,)
        
        # Create hold mask - fully vectorized
        hold_mask = torch.zeros(num_games, 5, device=device)
        
        # For each game, mark dice that match the best value
        best_values_expanded = best_value_indices.unsqueeze(1) + 1  # Convert to 1-6
        dice_matches = (dice_values == best_values_expanded).float()  # (num_games, 5)
        
        # Only hold if we have any unscored dice of this value
        has_unscored = (max_counts > 0).float().unsqueeze(1)
        hold_mask = dice_matches * has_unscored
        
        return hold_mask
    
    def decide_category(self, game):
        """
        Score the highest-scoring available category.
        Priority by expected value, tie-break by higher number.
        """
        num_games = game.num_games
        categories = torch.zeros(num_games, 1, dtype=torch.long, device=device)
        
        # Convert one-hot dice to values
        dice_values = torch.argmax(game.dice, dim=2) + 1  # (num_games, 5)
        
        for g in range(num_games):
            dice = dice_values[g]
            scores = game.scores[g]
            
            # Count occurrences of each die value
            counts = torch.zeros(7, device=device)
            for die in dice:
                counts[die] += 1
            
            best_category = -1
            best_score = -1
            
            # Check each category from highest value to lowest
            for cat in [5, 4, 3, 2, 1, 0]:  # 6s, 5s, 4s, 3s, 2s, 1s
                if scores[cat, 6] == 1:  # Unscored
                    die_value = cat + 1
                    score = counts[die_value] * die_value
                    
                    # Prefer categories where we have more dice
                    # But also consider the value
                    weighted_score = score
                    
                    # Bonus for having 3+ of a kind (likely to get bonus)
                    if counts[die_value] >= 3:
                        weighted_score += 5
                    
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_category = cat
            
            # If somehow no category is available (shouldn't happen), pick first unscored
            if best_category == -1:
                for cat in range(6):
                    if scores[cat, 6] == 1:
                        best_category = cat
                        break
            
            categories[g, 0] = best_category if best_category != -1 else 0
        
        return categories