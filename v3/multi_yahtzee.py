import torch
import torch.nn.functional as F

# Constants for scoring
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_VALUE = 35
SMALL_STRAIGHT_SCORE = 30
LARGE_STRAIGHT_SCORE = 40

# Small straight patterns (4 consecutive dice)
SMALL_STRAIGHT_PATTERNS = torch.tensor([
    [1, 1, 1, 1, 0, 0],  # 1-2-3-4
    [0, 1, 1, 1, 1, 0],  # 2-3-4-5
    [0, 0, 1, 1, 1, 1],  # 3-4-5-6
], dtype=torch.float32)

# Large straight patterns (5 consecutive dice)
LARGE_STRAIGHT_PATTERNS = torch.tensor([
    [1, 1, 1, 1, 1, 0],  # 1-2-3-4-5
    [0, 1, 1, 1, 1, 1],  # 2-3-4-5-6
], dtype=torch.float32)


class MultiYahtzee:
    """
    Multi-Yahtzee: A single game with Z scorecards and multipliers.
    
    Game structure:
    - ONE game with Z scorecards (not Z separate games)
    - All scorecards share the same dice rolls
    - Agent makes Z*13 total category selections (13 per scorecard)
    - Scorecard 0 has 1x multiplier, scorecard 1 has 2x, scorecard 2 has 3x, etc.
    - Each turn: roll dice 3 times, then select ONE category from ANY scorecard
    """
    
    def __init__(self, num_games, Z=3, device='cuda'):
        """
        Initialize Multi-Yahtzee game.
        
        Args:
            num_games: Number of parallel games (batch size)
            Z: Number of yahtzee scorecards per game
            device: Device to run on ('cuda' or 'cpu')
        """
        self.num_games = num_games
        self.Z = Z
        self.device = device
        self.total_rounds = Z * 13  # Total rounds in a complete multi-game
        
        # Dice state - shared across all Z games
        # One-hot encoded: [num_games, 5, 6]
        self.dice = torch.zeros(num_games, 5, 6, device=device)
        
        # Upper section scorecard - now with Z dimension
        # One-hot encoded for scores 0-6 for each die value (1-6)
        # Shape: [num_games, Z, 6, 7] (6 categories, 7 possible scores each 0-6)
        self.upper = torch.zeros(num_games, Z, 6, 7, device=device)
        # Initialize with last element = 1 to indicate unscored
        self.upper[:, :, :, 6] = 1
        
        # Lower section scorecard - now with Z dimension  
        # Shape: [num_games, Z, 7, 2]
        # Last dim: [points, unscored(1)/scored(0)]
        self.lower = torch.zeros(num_games, Z, 7, 2, device=device)
        self.lower[:, :, :, 1] = 1  # Initialize all as unscored
        
        # Turn tracker (which of 3 rolls we're on)
        # Shape: [num_games, Z, 3] (one-hot) - redundant Z dim since all Z games share dice
        self.turn = torch.zeros(num_games, Z, 3, device=device)
        
        # Round tracker (0 to Z*13-1)
        # Shape: [num_games]
        self.round = torch.zeros(num_games, dtype=torch.long, device=device)
        
        # Score multipliers for each game
        # Shape: [Z] - game 0 has multiplier 1, game 1 has 2, etc.
        self.multipliers = torch.arange(1, Z+1, device=device, dtype=torch.float32)
        
        # Track penalties (for invalid moves)
        self.penalties = torch.zeros(num_games, device=device)
        
    def clear(self):
        """Reset the game state."""
        self.dice.zero_()
        self.upper.zero_()
        self.upper[:, :, :, 6] = 1  # Set last element to 1 for unscored
        self.lower.zero_()
        self.lower[:, :, :, 1] = 1  # Set all as unscored
        self.turn.zero_()
        self.round.zero_()
        self.penalties.zero_()
        
    def roll_dice(self, hold_mask=None):
        """
        Roll dice with optional hold mask. Dice are sorted after rolling for consistency.
        
        Args:
            hold_mask: Binary tensor of shape [num_games, 5] indicating which dice to hold
        """
        if hold_mask is None:
            # Roll all dice
            roll = torch.randint(0, 6, (self.num_games, 5), device=self.device)
            # Sort the dice for consistency (helps neural network learn patterns)
            sorted_roll, _ = torch.sort(roll, dim=1)
            self.dice = F.one_hot(sorted_roll, num_classes=6).float()
        else:
            # Get current dice values from one-hot
            current_values = torch.argmax(self.dice, dim=2)  # [num_games, 5]
            
            # Roll new dice for unheld positions
            new_rolls = torch.randint(0, 6, (self.num_games, 5), device=self.device)
            
            # Combine held and new dice
            combined = torch.where(hold_mask.bool(), current_values, new_rolls)
            
            # Sort the combined dice for consistency
            sorted_values, _ = torch.sort(combined, dim=1)
            
            # Convert back to one-hot
            self.dice = F.one_hot(sorted_values, num_classes=6).float()
            
    def get_state(self):
        """
        Get current game state for neural network input.
        
        Returns:
            State tensor of shape [num_games, state_dim] where:
            state_dim = 30 (dice) + Z*6*7 (upper counts) + Z*7 (lower status) + 3 (turn)
                      = 33 + Z*49
        """
        # Flatten dice from [num_games, 5, 6] to [num_games, 30]
        dice_flat = self.dice.view(self.num_games, -1)
        
        # Flatten upper from [num_games, Z, 6, 7] to [num_games, Z*42]
        # Upper includes counts because bonus calculation needs them
        upper_flat = self.upper.view(self.num_games, -1)
        
        # For lower, only include scored/unscored status (index 1), not the actual scores
        # Extract just the status: [num_games, Z, 7, 1] -> [num_games, Z*7]
        lower_status = self.lower[:, :, :, 1]  # [num_games, Z, 7]
        lower_flat = lower_status.view(self.num_games, -1)  # [num_games, Z*7]
        
        # Turn is shared across all Z games - just take first Z's turn
        # Shape: [num_games, 3]
        turn_flat = self.turn[:, 0, :]  # All Z games have same turn
        
        # Concatenate all state components
        state = torch.cat([dice_flat, upper_flat, lower_flat, turn_flat], dim=-1)
        
        return state
        
    def get_available_categories(self):
        """
        Get mask of available categories across all Z games.
        
        Returns:
            Binary tensor of shape [num_games, Z*13] where 1 indicates available
        """
        # Check upper section availability (Z, 6 categories)
        # A category is available if the last element (index 6) is 1 (unscored marker)
        upper_available = self.upper[:, :, :, 6] == 1  # [num_games, Z, 6]
        
        # Check lower section availability (Z, 7 categories)
        # A category is available if unscored flag (index 1) is 1
        lower_available = self.lower[:, :, :, 1] == 1  # [num_games, Z, 7]
        
        # Combine upper and lower
        all_available = torch.cat([upper_available, lower_available], dim=-1)  # [num_games, Z, 13]
        
        # Flatten to [num_games, Z*13] for action selection
        available_flat = all_available.view(self.num_games, -1)
        
        return available_flat
        
    def calculate_upper_bonus(self):
        """
        Calculate the upper section bonus for all Z games.
        
        Returns:
            bonus: (num_games, Z) tensor of bonus values (0 or 35)
        """
        # Upper scores are in shape [num_games, Z, 6, 7]
        # Extract counts (indices 0-6 represent counts 0-6)
        # We need to convert one-hot back to actual scores
        
        # Create scoring matrix for each category
        # For category i, score = count * (i+1)
        score_values = torch.arange(7, device=self.device).float()  # [0, 1, 2, 3, 4, 5, 6]
        
        # Calculate scores for each category
        scores = torch.zeros(self.num_games, self.Z, 6, device=self.device)
        
        for cat in range(6):
            # Get the one-hot scores for this category
            cat_one_hot = self.upper[:, :, cat, :]  # [num_games, Z, 7]
            # Find which count is marked (argmax of one-hot)
            counts = torch.argmax(cat_one_hot, dim=-1).float()  # [num_games, Z]
            # Check if category is scored (last element = 0 means scored)
            is_scored = self.upper[:, :, cat, 6] == 0  # [num_games, Z]
            # Calculate score for this category
            scores[:, :, cat] = counts * (cat + 1) * is_scored.float()
        
        # Sum across categories to get total upper score
        upper_total = scores.sum(dim=-1)  # [num_games, Z]
        
        # Return 35 if >= 63, else 0
        bonus = torch.where(upper_total >= UPPER_BONUS_THRESHOLD,
                          torch.tensor(UPPER_BONUS_VALUE, dtype=torch.float32, device=self.device),
                          torch.tensor(0.0, dtype=torch.float32, device=self.device))
        
        return bonus
    
    def score_category(self, category_indices):
        """
        Score the selected category for each game.
        
        Args:
            category_indices: Tensor of shape [num_games] with values 0 to Z*13-1
                             indicating which category to score
                             
        Returns:
            Rewards tensor of shape [num_games] with (z+1) multiplier applied
        """
        # Initialize rewards
        rewards = torch.zeros(self.num_games, device=self.device)
        
        # With masking, invalid selections should not occur
        # We'll keep a simple check for debugging but no penalty
        flat_indices = category_indices.view(-1)  # [num_games]
        
        # Optional: Debug check (can be removed in production)
        # Uncomment to check for invalid selections during debugging
        # available = self.get_available_categories()  # [num_games, Z*13]
        # batch_indices = torch.arange(self.num_games, device=self.device)
        # selected_available = available[batch_indices, flat_indices]  # [num_games]
        # if not selected_available.all():
        #     # This should not happen with masking
        #     invalid_count = (~selected_available).sum().item()
        #     print(f"WARNING: {invalid_count} invalid category selections detected despite masking!")
        #     self.penalties += invalid_count
        
        # Use the selected indices directly
        final_indices = flat_indices
        
        # Calculate bonus before scoring
        bonus_before = self.calculate_upper_bonus()  # [num_games, Z]
        weighted_bonus_before = bonus_before * self.multipliers  # [num_games, Z]
        
        # Decode final category indices to (z_idx, cat_idx)
        z_idx = final_indices // 13  # [num_games]
        cat_idx = final_indices % 13  # [num_games]
        
        # Score upper section categories (0-5)
        upper_rewards = self._score_upper_internal(z_idx, cat_idx)
        rewards = rewards + upper_rewards
        
        # Calculate bonus after scoring  
        bonus_after = self.calculate_upper_bonus()  # [num_games, Z]
        weighted_bonus_after = bonus_after * self.multipliers  # [num_games, Z]
        
        # Add bonus difference (summed over Z)
        bonus_diff = (weighted_bonus_after - weighted_bonus_before).sum(dim=1)  # [num_games]
        rewards = rewards + bonus_diff
        
        # Score lower section categories (6-12)
        lower_rewards = self._score_lower_internal(z_idx, cat_idx)
        rewards = rewards + lower_rewards
        
        return rewards.unsqueeze(1)  # Return as [num_games, 1] for compatibility
    
    def _score_upper_internal(self, z_idx, cat_idx):
        """
        Internal method to score upper section categories.
        
        Args:
            z_idx: (num_games,) tensor of which Z game is being scored
            cat_idx: (num_games,) tensor of category indices (0-12)
        
        Returns:
            rewards: (num_games,) tensor of rewards with multiplier applied
        """
        # Mask for upper section categories (0-5)
        is_upper = (cat_idx < 6).float()  # [num_games]
        
        # Count dice values
        dice_counts = self.dice.sum(dim=1)  # [num_games, 6]
        
        # Broadcast dice counts to Z dimension
        dice_counts_z = dice_counts.unsqueeze(1).expand(-1, self.Z, -1)  # [num_games, Z, 6]
        
        # Convert dice counts to one-hot encoding for scores 0-6
        dice_counts_long = dice_counts_z.long()  # [num_games, Z, 6]
        dice_counts_one_hot = F.one_hot(dice_counts_long, num_classes=7).float()  # [num_games, Z, 6, 7]
        
        # Create category one-hot mask
        # This will be all zeros except for the selected (z, category) pair
        category_one_hot = torch.zeros(self.num_games, self.Z, 6, device=self.device)
        
        # Only set one-hot where upper section category is selected
        batch_indices = torch.arange(self.num_games, device=self.device)
        upper_mask = is_upper.bool()
        
        if upper_mask.any():
            # For games selecting upper categories, set the one-hot
            valid_batch = batch_indices[upper_mask]
            valid_z = z_idx[upper_mask]
            valid_cat = cat_idx[upper_mask].clamp(0, 5)  # Ensure in range 0-5
            category_one_hot[valid_batch, valid_z, valid_cat] = 1.0
        
        # Create masks for updating scores
        category_mask = category_one_hot.unsqueeze(-1)  # [num_games, Z, 6, 1]
        inverted_mask = 1 - category_mask  # [num_games, Z, 6, 1]
        
        # Update upper scores
        self.upper = inverted_mask * self.upper + category_mask * dice_counts_one_hot
        
        # Calculate base reward (before multiplier)
        # Only the selected category in the selected Z gets scored
        base_rewards = torch.zeros(self.num_games, device=self.device)
        
        if upper_mask.any():
            valid_batch = batch_indices[upper_mask]
            valid_cat = cat_idx[upper_mask].clamp(0, 5)
            
            # Get counts for the selected categories
            chosen_counts = torch.gather(dice_counts[upper_mask], 1, valid_cat.unsqueeze(1)).squeeze(1)
            
            # Calculate scores: count * (category + 1)
            die_values = valid_cat + 1
            scores = chosen_counts * die_values.float()
            
            # Apply the Z multiplier for the selected game
            valid_z = z_idx[upper_mask]
            multipliers_selected = self.multipliers[valid_z]
            
            base_rewards[upper_mask] = scores * multipliers_selected
        
        return base_rewards
    
    def _score_lower_internal(self, z_idx, cat_idx):
        """
        Internal method to score lower section categories.
        
        Args:
            z_idx: (num_games,) tensor of which Z game is being scored
            cat_idx: (num_games,) tensor of category indices (0-12)
        
        Returns:
            rewards: (num_games,) tensor of rewards with multiplier applied
        """
        # Calculate dice counts and sum
        dice_counts = self.dice.sum(dim=1)  # [num_games, 6]
        
        # Broadcast to Z dimension
        dice_counts_z = dice_counts.unsqueeze(1).expand(-1, self.Z, -1)  # [num_games, Z, 6]
        
        # Calculate dice sum
        face_values = torch.arange(1, 7, dtype=torch.float32, device=self.device)
        dice_sum = (dice_counts * face_values).sum(dim=1)  # [num_games]
        dice_sum_z = dice_sum.unsqueeze(1).expand(-1, self.Z)  # [num_games, Z]
        
        # Create scores tensor for all categories
        all_scores = torch.zeros(self.num_games, self.Z, 13, device=self.device)
        
        # 3-of-a-kind (category 6)
        has_3kind = (dice_counts_z >= 3).any(dim=2).float()  # [num_games, Z]
        all_scores[:, :, 6] = has_3kind * dice_sum_z
        
        # 4-of-a-kind (category 7)
        has_4kind = (dice_counts_z >= 4).any(dim=2).float()  # [num_games, Z]
        all_scores[:, :, 7] = has_4kind * dice_sum_z
        
        # Full house (category 8) - exactly one pair and one three-of-a-kind
        has_pair = (dice_counts_z == 2).any(dim=2).float()  # [num_games, Z]
        has_three = (dice_counts_z == 3).any(dim=2).float()  # [num_games, Z]
        has_full_house = has_pair * has_three
        all_scores[:, :, 8] = has_full_house * 25
        
        # Small straight (category 9) - 4 consecutive dice
        has_die = (dice_counts_z >= 1).float()  # [num_games, Z, 6]
        small_patterns = SMALL_STRAIGHT_PATTERNS.to(self.device)
        small_straight_check = torch.matmul(has_die, small_patterns.T)  # [num_games, Z, 3]
        has_small_straight = (small_straight_check >= 4).any(dim=2).float()  # [num_games, Z]
        all_scores[:, :, 9] = has_small_straight * SMALL_STRAIGHT_SCORE
        
        # Large straight (category 10) - 5 consecutive dice
        large_patterns = LARGE_STRAIGHT_PATTERNS.to(self.device)
        large_straight_check = torch.matmul(has_die, large_patterns.T)  # [num_games, Z, 2]
        has_large_straight = (large_straight_check >= 5).any(dim=2).float()  # [num_games, Z]
        all_scores[:, :, 10] = has_large_straight * LARGE_STRAIGHT_SCORE
        
        # Yahtzee (category 11) - 5 of a kind
        has_yahtzee = (dice_counts_z == 5).any(dim=2).float()  # [num_games, Z]
        all_scores[:, :, 11] = has_yahtzee * 50
        
        # Chance (category 12) - always sum of all dice
        all_scores[:, :, 12] = dice_sum_z
        
        # Create category selection mask
        # Only the selected (z, category) gets a 1
        category_mask = torch.zeros(self.num_games, self.Z, 13, device=self.device)
        batch_indices = torch.arange(self.num_games, device=self.device)
        
        # Mask for lower section categories (6-12)
        is_lower = (cat_idx >= 6) & (cat_idx < 13)
        lower_mask = is_lower.bool()
        
        if lower_mask.any():
            # For games selecting lower categories, set the mask
            valid_batch = batch_indices[lower_mask]
            valid_z = z_idx[lower_mask]
            valid_cat = cat_idx[lower_mask]
            category_mask[valid_batch, valid_z, valid_cat] = 1.0
        
        # Select scores using the mask
        selected_scores = (all_scores * category_mask).sum(dim=[1, 2])  # [num_games]
        
        # Apply multiplier for the selected Z game
        base_rewards = torch.zeros(self.num_games, device=self.device)
        
        if lower_mask.any():
            valid_z = z_idx[lower_mask]
            multipliers_selected = self.multipliers[valid_z]
            base_rewards[lower_mask] = selected_scores[lower_mask] * multipliers_selected
        
        # Update lower scorecard to mark categories as used and store scores
        if lower_mask.any():
            valid_batch = batch_indices[lower_mask]
            valid_z = z_idx[lower_mask]
            valid_cat = cat_idx[lower_mask] - 6  # Convert to 0-6 range for lower section
            
            # Store the RAW score (without multiplier) in index 0, mark as scored by setting index 1 to 0
            # This maintains consistency with upper section which stores counts not multiplied scores
            self.lower[valid_batch, valid_z, valid_cat, 0] = selected_scores[lower_mask]  # Store raw score
            self.lower[valid_batch, valid_z, valid_cat, 1] = 0.0  # Mark as scored
        
        return base_rewards
    
    def get_total_score(self):
        """
        Calculate total score for all games across all Z scorecards.
        
        Returns:
            Tensor of shape [num_games] with total scores
        """
        total = torch.zeros(self.num_games, device=self.device)
        
        # Add upper section scores
        for z in range(self.Z):
            for cat in range(6):
                # Check if scored (last element == 0)
                scored_mask = self.upper[:, z, cat, 6] == 0
                if scored_mask.any():
                    # Find the count for scored games
                    counts = torch.argmax(self.upper[:, z, cat, :6], dim=-1)
                    scores = counts * (cat + 1) * (z + 1)
                    total[scored_mask] += scores[scored_mask].float()
        
        # Add upper bonus
        bonus = self.calculate_upper_bonus()  # [num_games, Z]
        weighted_bonus = bonus * self.multipliers
        total += weighted_bonus.sum(dim=1)
        
        # Add lower section scores (now need to apply multiplier like upper section)
        for z in range(self.Z):
            # Check which categories are scored (index 1 == 0)
            scored_mask = self.lower[:, z, :, 1] == 0  # [num_games, 7]
            # Sum the scores for scored categories and apply multiplier
            raw_scores = (self.lower[:, z, :, 0] * scored_mask.float()).sum(dim=1)
            total += raw_scores * (z + 1)  # Apply the z+1 multiplier
        
        return total