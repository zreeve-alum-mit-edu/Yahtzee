import torch
from device_config import device


class RandomPlayer:
    def __init__(self):
        """
        Initialize RandomPlayer for making random game decisions.
        """
        pass
    
    def decide_hold(self, game):
        """
        Randomly decide which dice to hold.
        
        Args:
            game: SimpleYahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 5) with 0/1 values for hold decisions
        """
        # Randomly generate 0s and 1s for each die in each game
        hold_mask = torch.randint(0, 2, (game.num_games, 5), dtype=torch.float32, device=device)
        return hold_mask
    
    def decide_category(self, game):
        """
        Randomly decide which category to score.
        
        Args:
            game: SimpleYahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 1) with category index (0-5)
        """
        # Randomly generate category index from 0 to 5 for each game
        category = torch.randint(0, 6, (game.num_games, 1), dtype=torch.long, device=device)
        return category