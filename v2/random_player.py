import torch
from device_config import device


class RandomPlayer:
    def __init__(self):
        """
        Initialize RandomPlayer for making random game decisions.
        """
        self.training = False  # RandomPlayer is never in training mode
    
    def decide_hold(self, game):
        """
        Randomly decide which dice to hold.
        
        Args:
            game: Yahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 5) with 0/1 values for hold decisions
        """
        # Randomly generate 0s and 1s for each die in each game
        hold_mask = torch.randint(0, 2, (game.num_games, 5), dtype=torch.float32, device=device)
        return hold_mask
    
    def decide_category(self, game):
        """
        Randomly decide which category to score from available categories.
        
        Args:
            game: Yahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 1) with category index (0-12)
        """
        # Build mask of available categories for each game
        # Upper categories: unscored if upper_scores[..., 6] == 1
        upper_available = game.upper_scores[:, :, 6]  # Shape: (num_games, 6)
        
        # Lower categories: unscored if lower_scores[..., 1] == 1  
        lower_available = game.lower_scores[:, :, 1]  # Shape: (num_games, 7)
        
        # Combine into full availability mask
        available = torch.cat([upper_available, lower_available], dim=1).float()  # Shape: (num_games, 13)
        
        # Generate random values for each category
        random_values = torch.rand(game.num_games, 13, device=device)
        
        # Mask out unavailable categories with -inf so they won't be selected
        random_values = random_values * available - 1000 * (1 - available)
        
        # Select category with highest random value (which will be from available ones)
        category = torch.argmax(random_values, dim=1, keepdim=True)
        
        return category
    
    def get_state_from_game(self, game):
        """
        Extract state tensor from game (for compatibility with GameRunner).
        RandomPlayer doesn't actually use the state, but this is needed for the interface.
        
        Args:
            game: Yahtzee game instance
            
        Returns:
            Empty tensor (RandomPlayer doesn't use state)
        """
        # Return a dummy tensor - RandomPlayer doesn't use state for decisions
        return torch.zeros(game.num_games, 1, device=device)