import torch
from device_config import device


class SimpleYahtzee:
    def __init__(self, num_games):
        """
        Initialize SimpleYahtzee for parallel game processing.
        
        Args:
            num_games: Number of Yahtzee games to play in parallel (x dimension)
        """
        self.num_games = num_games
        self.device = device
        self.dice = torch.zeros(num_games, 5, 6, device=device)
        self.scores = torch.zeros(num_games, 6, 7, device=device)
        # Initialize scores with unscored pattern [0,0,0,0,0,0,1]
        self.scores[:, :, 6] = 1
        self.turn = torch.zeros(num_games, 3, device=device)
        self.penalties = torch.zeros(num_games, 1, device=device)
    
    def clear(self):
        """
        Clear all game state by zeroing out all tensors.
        """
        self.dice.zero_()
        self.scores.zero_()
        # Reset scores to unscored pattern [0,0,0,0,0,0,1]
        self.scores[:, :, 6] = 1
        self.turn.zero_()
        self.penalties.zero_()
    
    def roll_dice(self, hold_mask=None):
        """
        Roll dice, keeping held dice unchanged. Dice are sorted in ascending order.
        
        Args:
            hold_mask: Tensor of shape (num_games, 5) with 0/1 values indicating which dice to hold
        """
        # Generate random dice values (0-5) and convert to one-hot
        random_values = torch.randint(0, 6, (self.num_games, 5, 1), device=device)
        rolled_dice = torch.zeros(self.num_games, 5, 6, device=device)
        rolled_dice.scatter_(2, random_values, 1)
        
        if hold_mask is None:
            # Roll all dice
            self.dice = rolled_dice
        else:
            # Expand hold_mask from (x, 5) to (x, 5, 6)
            held = hold_mask.unsqueeze(-1).expand(-1, -1, 6)
            inverted_held = 1 - held
            
            # Update dice: keep held dice, roll unheld dice
            self.dice = (self.dice * held) + (rolled_dice * inverted_held)
        
        # Sort dice in ascending order
        # Convert one-hot to values for sorting
        dice_values = torch.argmax(self.dice, dim=2)  # (x, 5)
        sorted_values, _ = torch.sort(dice_values, dim=1)  # Sort along dice dimension
        
        # Convert back to one-hot
        sorted_dice = torch.zeros_like(self.dice)
        sorted_dice.scatter_(2, sorted_values.unsqueeze(-1), 1)
        self.dice = sorted_dice
    
    def score_category(self, category):
        """
        Score the selected category based on current dice.
        
        Args:
            category: Tensor of shape (num_games, 1) with category index (0-5)
        """
        # Check if chosen categories are already scored
        # Gather the 7th bit (index 6) for each game's chosen category
        chosen_scores = torch.gather(self.scores[:, :, 6], 1, category.long())  # (x, 1)
        # If 7th bit is 0, category is already scored (invalid)
        invalid_mask = (chosen_scores == 0).float()  # (x, 1)
        valid_mask = 1 - invalid_mask  # (x, 1)
        
        # Increment penalties for invalid choices
        self.penalties += invalid_mask
        
        # For invalid choices, find a random unscored category
        # Get unscored status for all categories (x, 6)
        unscored_categories = self.scores[:, :, 6]  # 1 if unscored, 0 if scored
        
        # Handle invalid games - sample from unscored categories
        # We need to handle this per game, so we'll use a different approach
        # Create replacement categories for all games (even valid ones)
        # Add small epsilon to avoid zero probabilities
        probs = unscored_categories + 1e-10
        replacement_categories = torch.multinomial(probs, 1)  # (x, 1)
        
        # Use mask to select original or replacement category
        final_category = (valid_mask * category + invalid_mask * replacement_categories).long()
        
        # Now proceed with scoring using final_category
        # Convert category index to one-hot encoding (x, 6)
        category_one_hot = torch.zeros(self.num_games, 6, device=device)
        category_one_hot.scatter_(1, final_category, 1)
        
        # Sum up dice values to get count of each die value (x, 6)
        dice_counts = self.dice.sum(dim=1)
        
        # Convert dice counts to one-hot encoding (x, 6, 7)
        dice_counts_one_hot = torch.zeros(self.num_games, 6, 7, device=device)
        # dice_counts contains values 0-5, representing counts of each die value
        # Convert to long and reshape for scatter
        dice_counts_expanded = dice_counts.long().unsqueeze(-1)
        dice_counts_one_hot.scatter_(2, dice_counts_expanded, 1)
        
        # Extend category_one_hot to (x, 6, 7)
        # Where category is selected: [1,1,1,1,1,1,0], otherwise all 0s
        category_mask = torch.zeros(self.num_games, 6, 7, device=device)
        # Create the pattern [1,1,1,1,1,1,0] and multiply by category_one_hot
        pattern = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.float32, device=device)
        category_mask = category_one_hot.unsqueeze(-1) * pattern
        
        # Create inverted category mask - all 1s where category is NOT selected, all 0s where selected
        inverted_category_mask = (1 - category_one_hot).unsqueeze(-1).expand(-1, -1, 7)
        
        # Update scores: keep existing scores for unselected categories, update selected category with dice counts
        self.scores = (inverted_category_mask * self.scores) + (category_mask * dice_counts_one_hot)
        
        # Calculate reward: count * (category + 1)
        # Use final_category for reward calculation
        chosen_counts = torch.gather(dice_counts, 1, final_category)  # (x, 1)
        die_values = final_category + 1  # (x, 1)
        reward = chosen_counts * die_values  # (x, 1)
        
        # Apply penalty of -50 for invalid choices (stronger signal)
        reward = reward - (invalid_mask * 50)
        
        return reward