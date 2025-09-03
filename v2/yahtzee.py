import torch
from device_config import device
from constants import (UPPER_SCORING_TENSOR, UPPER_BONUS_THRESHOLD, UPPER_BONUS_VALUE,
                       SMALL_STRAIGHT_PATTERNS, LARGE_STRAIGHT_PATTERNS, 
                       SMALL_STRAIGHT_SCORE, LARGE_STRAIGHT_SCORE)


class Yahtzee:
    """Full Yahtzee game with upper and lower sections using tensor operations."""
    
    def __init__(self, num_games):
        self.num_games = num_games
        self.device = device
        
        # Upper section scores: (num_games, 6, 7)
        # 6 categories: ones through sixes
        # 7 dims: [0 count, 1 count, 2 count, 3 count, 4 count, 5 count, unscored(1)/scored(0)]
        self.upper_scores = torch.zeros(num_games, 6, 7, device=device)
        self.upper_scores[:, :, 6] = 1  # All categories start unscored
        
        # Lower section scores: (num_games, 7, 2) 
        # 7 categories: 3-of-kind, 4-of-kind, full house, small straight, large straight, yahtzee, chance
        # 2 dims: [points, scored(0)/unscored(1)]
        self.lower_scores = torch.zeros(num_games, 7, 2, device=device)
        self.lower_scores[:, :, 1] = 1  # All categories start unscored
        
        # Dice: (num_games, 5, 6) - one-hot encoded
        self.dice = torch.zeros(num_games, 5, 6, device=device)
        
        # Turn tracker: (num_games, 3) - one-hot for which roll (1st, 2nd, 3rd)
        self.turn = torch.zeros(num_games, 3, device=device)
        
        # Round tracker: which round (0-12, since we have 13 categories total)
        self.round = torch.zeros(num_games, dtype=torch.long, device=device)
        
        # Penalties counter
        self.penalties = torch.zeros(num_games, 1, device=device)
    
    def clear(self):
        """Reset the game state for all games."""
        # Reset upper scores
        self.upper_scores[:] = 0
        self.upper_scores[:, :, 6] = 1  # All unscored
        
        # Reset lower scores  
        self.lower_scores[:] = 0
        self.lower_scores[:, :, 1] = 1  # All unscored
        
        # Reset dice
        self.dice[:] = 0
        
        # Reset turn
        self.turn[:] = 0
        
        # Reset round
        self.round[:] = 0
        
        # Reset penalties
        self.penalties[:] = 0
    
    def roll_dice(self, hold_mask=None):
        """Roll dice for all games, respecting hold masks if provided.
        
        Args:
            hold_mask: (num_games, 5) boolean tensor, True means hold that die
        """
        if hold_mask is None:
            # Roll all dice
            rolls = torch.randint(0, 6, (self.num_games, 5), device=device)
            # Sort the rolls
            rolls, _ = torch.sort(rolls, dim=1)
            self.dice = torch.nn.functional.one_hot(rolls, num_classes=6).float()
        else:
            # Get current dice values
            current_values = torch.argmax(self.dice, dim=2)
            
            # Convert hold_mask to boolean if it's float
            if hold_mask.dtype == torch.float32:
                hold_mask = hold_mask.bool()
            
            # Roll new dice for unhold positions
            roll_mask = ~hold_mask
            new_rolls = torch.randint(0, 6, (self.num_games, 5), device=device)
            
            # Combine held and new dice
            combined = torch.where(hold_mask, current_values, new_rolls)
            
            # Sort the combined dice
            sorted_values, _ = torch.sort(combined, dim=1)
            
            # Convert back to one-hot
            self.dice = torch.nn.functional.one_hot(sorted_values, num_classes=6).float()
    
    def calculate_upper_bonus(self, upper_scores=None):
        """Calculate the upper section bonus for all games.
        
        Args:
            upper_scores: Optional tensor to calculate bonus from. If None, uses self.upper_scores
            
        Returns:
            bonus: (num_games,) tensor of bonus values (0 or 35)
        """
        if upper_scores is None:
            upper_scores = self.upper_scores
            
        # Move scoring tensor to device
        scoring_tensor = UPPER_SCORING_TENSOR.to(device)
        
        # Get the count one-hot vectors for all categories
        count_one_hots = upper_scores[:, :, :6]  # (num_games, 6, 6)
        
        # Multiply by scoring tensor to get scores
        # scoring_tensor is (6, 6), count_one_hots is (num_games, 6, 6)
        # We want element-wise multiply along the category dimension
        scores_per_category = count_one_hots * scoring_tensor.unsqueeze(0)  # (num_games, 6, 6)
        
        # Sum across the count dimension to get score per category
        scores_per_category = scores_per_category.sum(dim=2)  # (num_games, 6)
        
        # Mask by which categories are scored (status = 0)
        is_scored = (upper_scores[:, :, 6] == 0).float()  # (num_games, 6)
        scores_per_category = scores_per_category * is_scored
        
        # Sum to get total upper score per game
        upper_total = scores_per_category.sum(dim=1)  # (num_games,)
        
        # Calculate bonus (35 if >= 63, else 0)
        bonus = torch.where(upper_total >= UPPER_BONUS_THRESHOLD,
                          torch.tensor(UPPER_BONUS_VALUE, dtype=torch.float32, device=device),
                          torch.tensor(0.0, device=device))
        
        return bonus
    
    def score_category(self, category):
        """Score a category for all games, handling validation and penalties.
        
        Args:
            category: (num_games, 1) tensor of category indices (0-12)
                      0-5 are upper section, 6-12 are lower section
        
        Returns:
            rewards: (num_games, 1) tensor of total rewards
        """
        # Initialize reward tensor with zeros
        rewards = torch.zeros(self.num_games, 1, device=device)
        
        # Check which categories are already scored
        # Upper section: categories 0-5
        is_upper = (category < 6)
        is_lower = (category >= 6) & (category < 13)
        
        # Initialize final_category as the requested category
        final_category = category.clone()
        
        # Build mask for games needing penalties (already scored categories)
        penalty_mask = torch.zeros(self.num_games, dtype=torch.bool, device=device)
        
        # Check upper section for already scored categories
        upper_mask = is_upper.squeeze(-1)
        if upper_mask.any():
            upper_indices = upper_mask.nonzero(as_tuple=True)[0]
            upper_cats = category[upper_indices].clamp(0, 5)
            
            # Check scored status
            upper_status = torch.gather(self.upper_scores[upper_indices, :, 6], 1, upper_cats)
            already_scored_upper = (upper_status == 0).squeeze(-1)
            
            # Mark games that need penalties
            if already_scored_upper.any():
                penalty_indices = upper_indices[already_scored_upper]
                penalty_mask[penalty_indices] = True
        
        # Check lower section for already scored categories
        lower_mask = is_lower.squeeze(-1)
        if lower_mask.any():
            lower_indices = lower_mask.nonzero(as_tuple=True)[0]
            lower_cats = (category[lower_indices] - 6).clamp(0, 6)
            
            # Check scored status
            lower_status = torch.gather(self.lower_scores[lower_indices, :, 1], 1, lower_cats)
            already_scored_lower = (lower_status == 0).squeeze(-1)
            
            # Mark games that need penalties
            if already_scored_lower.any():
                penalty_indices = lower_indices[already_scored_lower]
                penalty_mask[penalty_indices] = True
        
        # Vectorized penalty application and category reassignment
        if penalty_mask.any():
            # Apply penalties
            self.penalties[penalty_mask] += 1
            rewards[penalty_mask] -= 10
            
            # Build unscored mask for all penalized games
            upper_unscored = self.upper_scores[:, :, 6]  # (num_games, 6)
            lower_unscored = self.lower_scores[:, :, 1]  # (num_games, 7)
            all_unscored = torch.cat([upper_unscored, lower_unscored], dim=1)  # (num_games, 13)
            
            # Sample new categories for penalized games using batched multinomial
            penalized_unscored = all_unscored[penalty_mask]  # (num_penalized, 13)
            probs = penalized_unscored + 1e-10
            new_cats = torch.multinomial(probs, 1)  # (num_penalized, 1)
            
            # Update final categories for penalized games
            final_category[penalty_mask] = new_cats
        
        # Now score using final_category
        # Call upper scoring for categories 0-5
        upper_rewards = self._score_upper_internal(final_category)
        rewards += upper_rewards
        
        # Call lower scoring for categories 6-12
        lower_rewards = self._score_lower_internal(final_category)
        rewards += lower_rewards
        
        return rewards
    
    def _score_upper_internal(self, category):
        """Internal method to score upper section categories.
        Assumes category validation already done.
        
        Args:
            category: (num_games, 1) tensor of category indices (0-12)
        
        Returns:
            rewards: (num_games, 1) tensor of rewards for upper section scoring
        """
        # Calculate bonus before any updates
        bonus_before = self.calculate_upper_bonus()  # (num_games,)
        
        # Mask for upper section categories (0-5)
        is_upper = (category < 6).float()  # (num_games, 1)
        
        # Clamp to valid range for upper section
        safe_category = torch.clamp(category, 0, 5)
        
        # Convert category index to one-hot encoding (num_games, 6)
        category_one_hot = torch.zeros(self.num_games, 6, device=device)
        # Only scatter where is_upper = 1, otherwise category_one_hot stays all zeros
        scatter_indices = torch.where(is_upper > 0, safe_category, torch.zeros_like(safe_category))
        category_one_hot.scatter_(1, scatter_indices, is_upper)
        
        # Sum up dice values to get count of each die value (num_games, 6)
        dice_counts = self.dice.sum(dim=1)
        
        # Convert dice counts to one-hot encoding (num_games, 6, 7)
        dice_counts_one_hot = torch.zeros(self.num_games, 6, 7, device=device)
        dice_counts_expanded = dice_counts.long().unsqueeze(-1)
        dice_counts_one_hot.scatter_(2, dice_counts_expanded, 1)
        
        # Create category mask for updating scores
        pattern = torch.tensor([1, 1, 1, 1, 1, 1, 0], dtype=torch.float32, device=device)
        category_mask = category_one_hot.unsqueeze(-1) * pattern  # All zeros if lower section
        
        # Create inverted category mask
        inverted_category_mask = (1 - category_one_hot).unsqueeze(-1).expand(-1, -1, 7)
        
        # Update scores: keep existing scores for unselected categories, update selected category
        # If lower section, category_mask is all zeros so nothing changes
        self.upper_scores = (inverted_category_mask * self.upper_scores) + (category_mask * dice_counts_one_hot)
        
        # Calculate reward: count * (category + 1)
        chosen_counts = torch.gather(dice_counts, 1, safe_category)  # (num_games, 1)
        die_values = safe_category + 1  # (num_games, 1)
        reward = chosen_counts * die_values * is_upper  # Zero if lower section
        
        # Calculate bonus after updates
        bonus_after = self.calculate_upper_bonus()  # (num_games,)
        
        # Add bonus difference to reward (only rewards bonus once when threshold is crossed)
        bonus_diff = (bonus_after - bonus_before).unsqueeze(-1)  # (num_games, 1)
        reward = reward + bonus_diff
        
        return reward
    
    def _calculate_dice_counts(self):
        """Calculate count of each die value for all games.
        
        Returns:
            dice_counts: (num_games, 6) tensor with counts of each die value
            dice_sum: (num_games,) tensor with sum of all dice values
        """
        # Dice is already one-hot encoded: (num_games, 5, 6)
        # Sum across the dice dimension to get counts directly
        dice_counts = self.dice.sum(dim=1)  # (num_games, 6)
        
        # Calculate sum of dice values
        # Multiply counts by face values (1-6) and sum
        face_values = torch.arange(1, 7, dtype=torch.float32, device=device)
        dice_sum = (dice_counts * face_values).sum(dim=1)  # (num_games,)
        
        return dice_counts, dice_sum
    
    def _score_n_of_kind(self, dice_counts, dice_sum, n):
        """Score n-of-a-kind categories (3-of-kind, 4-of-kind).
        
        Args:
            dice_counts: (num_games, 6) tensor with counts of each die value
            dice_sum: (num_games,) tensor with sum of all dice values
            n: Number of same dice required (3 or 4)
        
        Returns:
            scores: (num_games,) tensor of scores for this category
        """
        # Check if any die value has count >= n
        has_n_of_kind = (dice_counts >= n).any(dim=1).float()  # (num_games,)
        
        # Score is sum of all dice if condition is met, else 0
        scores = has_n_of_kind * dice_sum
        
        return scores
    
    def _score_lower_internal(self, category):
        """Internal method to score lower section categories.
        Assumes category validation already done.
        
        Args:
            category: (num_games, 1) tensor of category indices (0-12)
        
        Returns:
            rewards: (num_games, 1) tensor of rewards for lower section scoring
        """
        # Calculate dice counts and sum
        dice_counts, dice_sum = self._calculate_dice_counts()
        
        # Create scores tensor for all categories (num_games, 13)
        all_scores = torch.zeros(self.num_games, 13, device=device)
        
        # Lower section categories:
        # 6: 3-of-a-kind
        # 7: 4-of-a-kind
        # 8: Full house (25 points)
        # 9: Small straight (30 points)
        # 10: Large straight (40 points)
        # 11: Yahtzee (50 points)
        # 12: Chance (sum of all dice)
        
        # 3-of-a-kind (category 6)
        all_scores[:, 6] = self._score_n_of_kind(dice_counts, dice_sum, 3)
        
        # 4-of-a-kind (category 7)
        all_scores[:, 7] = self._score_n_of_kind(dice_counts, dice_sum, 4)
        
        # Full house (category 8) - exactly one pair and one three-of-a-kind
        has_pair = (dice_counts == 2).any(dim=1).float()  # (num_games,)
        has_three = (dice_counts == 3).any(dim=1).float()  # (num_games,)
        has_full_house = has_pair * has_three  # Both conditions must be true
        all_scores[:, 8] = has_full_house * 25
        
        # Small straight (category 9) - 4 consecutive dice
        # Convert dice_counts to binary (has at least 1 of each die value)
        has_die = (dice_counts >= 1).float()  # (num_games, 6)
        
        # Check each small straight pattern
        # Multiply pattern with has_die and sum - if sum >= 4, we have a small straight
        small_straight_check = torch.matmul(has_die, SMALL_STRAIGHT_PATTERNS.T.to(device))  # (num_games, 3)
        has_small_straight = (small_straight_check >= 4).any(dim=1).float()  # (num_games,)
        all_scores[:, 9] = has_small_straight * SMALL_STRAIGHT_SCORE
        
        # Large straight (category 10) - 5 consecutive dice
        # Check each large straight pattern
        large_straight_check = torch.matmul(has_die, LARGE_STRAIGHT_PATTERNS.T.to(device))  # (num_games, 2)
        has_large_straight = (large_straight_check >= 5).any(dim=1).float()  # (num_games,)
        all_scores[:, 10] = has_large_straight * LARGE_STRAIGHT_SCORE
        
        # Yahtzee (category 11) - 5 of a kind
        has_yahtzee = (dice_counts == 5).any(dim=1).float()  # (num_games,)
        all_scores[:, 11] = has_yahtzee * 50
        
        # Chance (category 12) - always sum of all dice
        all_scores[:, 12] = dice_sum
        
        # Gather scores for the selected categories
        rewards = torch.gather(all_scores, 1, category)  # (num_games, 1)
        
        # Mask for lower section categories (6-12)
        is_lower = ((category >= 6) & (category < 13)).float()  # (num_games, 1)
        
        # Apply mask (only score if it's a lower section category)
        rewards = rewards * is_lower
        
        # Update lower_scores tensor to mark categories as scored using vectorized operations
        mask = (is_lower.squeeze(-1) > 0)  # (num_games,) boolean mask
        idx = torch.nonzero(mask, as_tuple=True)[0]  # indices of games scoring lower
        
        if idx.numel() > 0:
            # Convert category to 0-6 range for lower section
            lower_cat = (category - 6).clamp(0, 6)
            cats = lower_cat[idx].squeeze(-1)  # (N,) category indices for selected games
            
            # Vectorized update: mark as scored and set points
            self.lower_scores[idx, cats, 1] = 0  # unscored flag -> 0
            self.lower_scores[idx, cats, 0] = rewards[idx, 0]  # set points
        
        return rewards
    
    def score_upper(self, category):
        """Score an upper section category for all games.
        Provided for backward compatibility with tests.
        
        Args:
            category: (num_games, 1) tensor of category indices (0-12)
        
        Returns:
            rewards: (num_games, 1) tensor of rewards
        """
        return self.score_category(category)