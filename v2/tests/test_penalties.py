import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from yahtzee import Yahtzee


class TestPenalties(unittest.TestCase):
    
    def test_penalties_tensor_initialized(self):
        """Test that penalties tensor is properly initialized."""
        game = Yahtzee(5)
        
        self.assertEqual(game.penalties.shape, (5, 1), "Penalties should have shape (num_games, 1)")
        self.assertTrue(torch.all(game.penalties == 0), "Penalties should be initialized to 0")
    
    def test_clear_resets_penalties(self):
        """Test that clear() resets penalties to zero."""
        game = Yahtzee(3)
        
        # Manually set some penalties
        game.penalties[:] = 5
        
        # Clear should reset
        game.clear()
        
        self.assertTrue(torch.all(game.penalties == 0), "Penalties should be reset to 0 after clear()")
    
    def test_scoring_already_scored_upper_category_penalty(self):
        """Test that choosing an already scored upper category incurs a penalty."""
        game = Yahtzee(1)
        
        # Set up dice
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 2] = 1  # All threes
        
        # Score category 2 (threes) first time - should work normally
        category = torch.tensor([[2]], dtype=torch.long, device=game.device)
        reward1 = game.score_upper(category)
        
        # Check no penalty yet
        self.assertEqual(game.penalties[0, 0].item(), 0, "No penalty for first scoring")
        self.assertEqual(reward1[0, 0].item(), 15, "Reward should be 5 threes * 3 = 15")
        
        # Try to score same category again
        reward2 = game.score_upper(category)
        
        # Check penalty was applied
        self.assertEqual(game.penalties[0, 0].item(), 1, "Penalty counter should increment")
        # Reward depends on which random category is selected
        # With 5 threes, could be:
        # - Chance: 15 - 10 = 5
        # - 3-of-kind: 15 - 10 = 5  
        # - 4-of-kind: 15 - 10 = 5
        # - Other upper categories: 0 - 10 = -10
        # So we just verify penalty was applied, not the exact reward value
    
    def test_multiple_penalties_accumulate(self):
        """Test that multiple penalties accumulate correctly."""
        game = Yahtzee(1)
        
        # Set up dice
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, 0:3, 0] = 1  # 3 ones
        game.dice[0, 3:, 1] = 1    # 2 twos
        
        # Score category 0 (ones)
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward1 = game.score_upper(category)
        self.assertEqual(game.penalties[0, 0].item(), 0)
        self.assertEqual(reward1[0, 0].item(), 3, "3 ones * 1 = 3")
        
        # Try to score it again
        reward2 = game.score_upper(category)
        self.assertEqual(game.penalties[0, 0].item(), 1, "First penalty")
        
        # Try to score it yet again
        reward3 = game.score_upper(category)
        self.assertEqual(game.penalties[0, 0].item(), 2, "Second penalty accumulated")
        
        # Invalid attempts should give reward - 10 (where reward is from random category)
        # With the dice (3 ones, 2 twos), this is a full house, which could score 25
        # So the penalty reward could be as high as 25 - 10 = 15
        # We just check that penalties were applied
        self.assertEqual(game.penalties[0, 0].item(), 2, "Should have 2 penalties total")
    
    def test_penalty_chooses_random_unscored_category(self):
        """Test that when penalized, a random unscored category is scored instead."""
        game = Yahtzee(1)
        
        # Score several categories first
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        
        # Score categories 0, 1, 2, 3
        for cat in range(4):
            game.dice[:] = 0
            game.dice[0, :, cat] = 1  # All of that die value
            category = torch.tensor([[cat]], dtype=torch.long, device=game.device)
            game.score_upper(category)
        
        # Now we have 2 unscored categories (4 and 5)
        # Set dice to have some 5s and 6s
        game.dice[:] = 0
        game.dice[0, 0:2, 4] = 1  # 2 fives
        game.dice[0, 2:, 5] = 1    # 3 sixes
        
        # Try to score already-scored category 0
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Check that some category was scored (either upper 4/5 or any lower)
        cat4_scored = (game.upper_scores[0, 4, 6] == 0).item()  # 7th bit is 0 means scored
        cat5_scored = (game.upper_scores[0, 5, 6] == 0).item()
        any_lower_scored = (game.lower_scores[0, :, 1] == 0).any().item()
        
        self.assertTrue(cat4_scored or cat5_scored or any_lower_scored, 
                       "One of the unscored categories should have been scored")
        
        # Original category 0 should still have its original score
        self.assertEqual(game.upper_scores[0, 0, 6].item(), 0, "Category 0 should remain scored")
    
    def test_multiple_games_penalties_independent(self):
        """Test that penalties are tracked independently for each game."""
        game = Yahtzee(3)
        
        # Set up dice for all games
        game.dice = torch.zeros(3, 5, 6, device=game.device)
        game.dice[:, :, 1] = 1  # All twos for all games
        
        # Score category 1 for all games
        category = torch.tensor([[1], [1], [1]], dtype=torch.long, device=game.device)
        reward1 = game.score_upper(category)
        
        # All should have normal rewards, no penalties
        self.assertTrue(torch.all(game.penalties == 0))
        self.assertTrue(torch.all(reward1 == 10))  # 5 twos * 2 = 10
        
        # Now have different games try different things
        # Game 0: try already-scored category (penalty)
        # Game 1: try unscored category (no penalty)  
        # Game 2: try already-scored category (penalty)
        category = torch.tensor([[1], [2], [1]], dtype=torch.long, device=game.device)
        reward2 = game.score_upper(category)
        
        # Check penalties
        self.assertEqual(game.penalties[0, 0].item(), 1, "Game 0 should have penalty")
        self.assertEqual(game.penalties[1, 0].item(), 0, "Game 1 should have no penalty")
        self.assertEqual(game.penalties[2, 0].item(), 1, "Game 2 should have penalty")
        
        # Check rewards
        # Penalty rewards should be <= 0 (could be 0 if chance is picked and scores 10)
        self.assertLessEqual(reward2[0, 0].item(), 0, "Game 0 penalty reward")
        self.assertEqual(reward2[1, 0].item(), 0, "Game 1: 0 threes * 3 = 0")
        self.assertLessEqual(reward2[2, 0].item(), 0, "Game 2 penalty reward")
    
    def test_penalty_when_all_upper_categories_scored(self):
        """Test edge case when all upper categories are already scored."""
        game = Yahtzee(1)
        
        # Score all 6 upper categories
        for cat in range(6):
            game.dice = torch.zeros(1, 5, 6, device=game.device)
            game.dice[0, :, cat] = 1
            category = torch.tensor([[cat]], dtype=torch.long, device=game.device)
            game.score_upper(category)
        
        # All categories should be scored now
        all_scored = torch.all(game.upper_scores[0, :, 6] == 0)
        self.assertTrue(all_scored, "All categories should be scored")
        
        # Try to score again - should still handle it gracefully
        game.dice[:] = 0
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should get penalty
        self.assertEqual(game.penalties[0, 0].item(), 1)
        # Reward should be negative
        self.assertLess(reward[0, 0].item(), 0, "Reward should be negative with penalty")
    
    def test_penalty_subtracts_exactly_10(self):
        """Test that penalty subtracts exactly 10 from whatever score the replacement category gets."""
        game = Yahtzee(1)
        
        # Score category 0 first
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        game.score_upper(category)
        
        # Set specific dice
        game.dice[:] = 0
        game.dice[0, 0:2, 2] = 1  # 2 threes
        game.dice[0, 2:, 3] = 1    # 3 fours
        
        # Try to score category 0 again (already scored)
        # This will randomly pick an unscored category and score it
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        
        # Save the current scores state
        scores_before = game.upper_scores.clone()
        
        reward = game.score_upper(category)
        
        # Find which category actually got scored (changed from unscored to scored)
        # Check both upper and lower sections
        upper_scored_mask = (scores_before[:, :, 6] == 1) & (game.upper_scores[:, :, 6] == 0)
        lower_before = torch.ones(1, 7, device=game.device)  # All lower were unscored initially
        lower_scored_mask = (lower_before == 1) & (game.lower_scores[:, :, 1] == 0)
        
        # Determine which category was scored
        if upper_scored_mask[0].any():
            scored_category = upper_scored_mask[0].nonzero()[0].item()
            is_upper_section = True
        elif lower_scored_mask[0].any():
            scored_category = lower_scored_mask[0].nonzero()[0].item() + 6  # Add 6 for lower section
            is_upper_section = False
        else:
            # Nothing was scored (shouldn't happen)
            self.fail("No category was scored")
        
        # Calculate what the reward should have been without penalty
        if is_upper_section:
            if scored_category == 2:  # threes
                expected_base = 2 * 3  # 2 threes * 3
            elif scored_category == 3:  # fours
                expected_base = 3 * 4  # 3 fours * 4
            else:
                # Other categories have 0 count
                expected_base = 0
        else:
            # Lower section scoring with 2 threes and 3 fours
            if scored_category == 6:  # 3-of-kind
                expected_base = 18  # Sum of dice
            elif scored_category == 7:  # 4-of-kind
                expected_base = 0   # Doesn't qualify
            elif scored_category == 8:  # Full house
                expected_base = 25  # 2+3 pattern qualifies
            elif scored_category == 12:  # Chance
                expected_base = 18  # Sum of dice
            else:
                expected_base = 0
        
        # Reward should be base score minus 10
        self.assertEqual(reward[0, 0].item(), expected_base - 10,
                        f"Reward should be {expected_base} - 10 = {expected_base - 10}")
    
    def test_lower_category_no_penalty_from_upper(self):
        """Test that choosing a lower section category doesn't trigger upper section penalties."""
        game = Yahtzee(1)
        
        # Set up dice
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 2] = 1  # All threes (5 threes)
        
        # Try to score lower section category (e.g., 7 = 4-of-kind)
        category = torch.tensor([[7]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should not get any penalty
        self.assertEqual(game.penalties[0, 0].item(), 0, "No penalty for lower section category")
        # Reward should be 15 (5 threes qualify for 4-of-kind, sum = 15)
        self.assertEqual(reward[0, 0].item(), 15, "Reward should be 15 for 4-of-kind with 5 threes")
        
        # Upper scores should be unchanged
        self.assertTrue(torch.all(game.upper_scores[0, :, 6] == 1), "All upper categories should remain unscored")
        
        # Lower section category 7 (4-of-kind) should be scored
        self.assertEqual(game.lower_scores[0, 1, 1].item(), 0, "4-of-kind should be marked as scored")
        self.assertEqual(game.lower_scores[0, 1, 0].item(), 15, "4-of-kind should have 15 points")


if __name__ == '__main__':
    unittest.main()