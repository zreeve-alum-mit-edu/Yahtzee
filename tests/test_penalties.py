import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from simple_yahtzee import SimpleYahtzee


class TestPenalties(unittest.TestCase):
    
    def test_penalties_tensor_initialized(self):
        """Test that penalties tensor is properly initialized."""
        game = SimpleYahtzee(5)
        
        self.assertEqual(game.penalties.shape, (5, 1), "Penalties should have shape (num_games, 1)")
        self.assertTrue(torch.all(game.penalties == 0), "Penalties should be initialized to 0")
    
    def test_clear_resets_penalties(self):
        """Test that clear() resets penalties to zero."""
        game = SimpleYahtzee(3)
        
        # Manually set some penalties
        game.penalties[:] = 5
        
        # Clear should reset
        game.clear()
        
        self.assertTrue(torch.all(game.penalties == 0), "Penalties should be reset to 0 after clear()")
    
    def test_scoring_already_scored_category_penalty(self):
        """Test that choosing an already scored category incurs a penalty."""
        game = SimpleYahtzee(1)
        
        # Set up dice
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, :, 2] = 1  # All threes
        
        # Score category 2 (threes) first time - should work normally
        category = torch.tensor([[2]], dtype=torch.long)
        reward1 = game.score_category(category)
        
        # Check no penalty yet
        self.assertEqual(game.penalties[0, 0].item(), 0, "No penalty for first scoring")
        self.assertEqual(reward1[0, 0].item(), 15, "Reward should be 5 threes * 3 = 15")
        
        # Try to score same category again
        reward2 = game.score_category(category)
        
        # Check penalty was applied
        self.assertEqual(game.penalties[0, 0].item(), 1, "Penalty counter should increment")
        self.assertEqual(reward2[0, 0].item(), -10, "Reward should be -10 for invalid choice")
    
    def test_multiple_penalties_accumulate(self):
        """Test that multiple penalties accumulate correctly."""
        game = SimpleYahtzee(1)
        
        # Set up dice
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, 0:3, 0] = 1  # 3 ones
        game.dice[0, 3:, 1] = 1    # 2 twos
        
        # Score category 0 (ones)
        category = torch.tensor([[0]], dtype=torch.long)
        reward1 = game.score_category(category)
        self.assertEqual(game.penalties[0, 0].item(), 0)
        self.assertEqual(reward1[0, 0].item(), 3, "3 ones * 1 = 3")
        
        # Try to score it again
        reward2 = game.score_category(category)
        self.assertEqual(game.penalties[0, 0].item(), 1, "First penalty")
        
        # Try to score it yet again
        reward3 = game.score_category(category)
        self.assertEqual(game.penalties[0, 0].item(), 2, "Second penalty accumulated")
        
        # Invalid attempts should give reward - 10 (where reward is from random category)
        # Since we have 2 twos, if category 1 gets randomly selected, it's 2*2=4, minus 10 = -6
        # The reward will be the score from whatever random category was chosen, minus 10
        self.assertLessEqual(reward2[0, 0].item(), -10 + 15)  # At most 5*6-10=20
        self.assertLessEqual(reward3[0, 0].item(), -10 + 15)  # At most 5*6-10=20
    
    def test_penalty_chooses_random_unscored_category(self):
        """Test that when penalized, a random unscored category is scored instead."""
        game = SimpleYahtzee(1)
        
        # Score several categories first
        game.dice = torch.zeros(1, 5, 6)
        
        # Score categories 0, 1, 2, 3
        for cat in range(4):
            game.dice[:] = 0
            game.dice[0, :, cat] = 1  # All of that die value
            category = torch.tensor([[cat]], dtype=torch.long)
            game.score_category(category)
        
        # Now we have 2 unscored categories (4 and 5)
        # Set dice to have some 5s and 6s
        game.dice[:] = 0
        game.dice[0, 0:2, 4] = 1  # 2 fives
        game.dice[0, 2:, 5] = 1    # 3 sixes
        
        # Try to score already-scored category 0
        category = torch.tensor([[0]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Check that either category 4 or 5 was scored (not 0)
        cat4_scored = (game.scores[0, 4, 6] == 0)  # 7th bit is 0 means scored
        cat5_scored = (game.scores[0, 5, 6] == 0)
        
        self.assertTrue(cat4_scored or cat5_scored, 
                       "One of the unscored categories should have been scored")
        
        # Original category 0 should still have its original score
        self.assertEqual(game.scores[0, 0, 6].item(), 0, "Category 0 should remain scored")
    
    def test_multiple_games_penalties_independent(self):
        """Test that penalties are tracked independently for each game."""
        game = SimpleYahtzee(3)
        
        # Set up dice for all games
        game.dice = torch.zeros(3, 5, 6)
        game.dice[:, :, 1] = 1  # All twos for all games
        
        # Score category 1 for all games
        category = torch.tensor([[1], [1], [1]], dtype=torch.long)
        reward1 = game.score_category(category)
        
        # All should have normal rewards, no penalties
        self.assertTrue(torch.all(game.penalties == 0))
        self.assertTrue(torch.all(reward1 == 10))  # 5 twos * 2 = 10
        
        # Now have different games try different things
        # Game 0: try already-scored category (penalty)
        # Game 1: try unscored category (no penalty)  
        # Game 2: try already-scored category (penalty)
        category = torch.tensor([[1], [2], [1]], dtype=torch.long)
        reward2 = game.score_category(category)
        
        # Check penalties
        self.assertEqual(game.penalties[0, 0].item(), 1, "Game 0 should have penalty")
        self.assertEqual(game.penalties[1, 0].item(), 0, "Game 1 should have no penalty")
        self.assertEqual(game.penalties[2, 0].item(), 1, "Game 2 should have penalty")
        
        # Check rewards
        self.assertEqual(reward2[0, 0].item(), -10, "Game 0 penalty reward")
        self.assertEqual(reward2[1, 0].item(), 0, "Game 1: 0 threes * 3 = 0")
        self.assertEqual(reward2[2, 0].item(), -10, "Game 2 penalty reward")
    
    def test_penalty_when_all_categories_scored(self):
        """Test edge case when all categories are already scored."""
        game = SimpleYahtzee(1)
        
        # Score all 6 categories
        for cat in range(6):
            game.dice = torch.zeros(1, 5, 6)
            game.dice[0, :, cat] = 1
            category = torch.tensor([[cat]], dtype=torch.long)
            game.score_category(category)
        
        # All categories should be scored now
        all_scored = torch.all(game.scores[0, :, 6] == 0)
        self.assertTrue(all_scored, "All categories should be scored")
        
        # Try to score again - should still handle it gracefully
        game.dice[:] = 0
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Should get penalty
        self.assertEqual(game.penalties[0, 0].item(), 1)
        # The reward calculation might vary since all categories are scored
        # but we should definitely get the -10 penalty applied
        self.assertLessEqual(reward[0, 0].item(), -10 + 5, 
                            "Reward should have -10 penalty applied")
    
    def test_penalty_subtracts_exactly_10(self):
        """Test that penalty subtracts exactly 10 from whatever score the replacement category gets."""
        game = SimpleYahtzee(1)
        
        # Score category 0 first
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long)
        game.score_category(category)
        
        # Set specific dice
        game.dice[:] = 0
        game.dice[0, 0:2, 2] = 1  # 2 threes
        game.dice[0, 2:, 3] = 1    # 3 fours
        
        # Try to score category 0 again (already scored)
        # This will randomly pick an unscored category and score it
        category = torch.tensor([[0]], dtype=torch.long)
        
        # To verify the -10 penalty, we need to check what actually got scored
        # Save the current scores state
        scores_before = game.scores.clone()
        
        reward = game.score_category(category)
        
        # Find which category actually got scored (changed from unscored to scored)
        scored_mask = (scores_before[:, :, 6] == 1) & (game.scores[:, :, 6] == 0)
        scored_category = scored_mask[0].nonzero()[0].item()
        
        # Calculate what the reward should have been without penalty
        if scored_category == 2:  # threes
            expected_base = 2 * 3  # 2 threes * 3
        elif scored_category == 3:  # fours
            expected_base = 3 * 4  # 3 fours * 4
        else:
            # Other categories have 0 count
            expected_base = 0
        
        # Reward should be base score minus 10
        self.assertEqual(reward[0, 0].item(), expected_base - 10,
                        f"Reward should be {expected_base} - 10 = {expected_base - 10}")
    
    def test_penalty_reward_calculation(self):
        """Test that penalty correctly subtracts 10 from the normal reward."""
        game = SimpleYahtzee(2)
        
        # Set up same dice for both games
        game.dice = torch.zeros(2, 5, 6)
        game.dice[:, 0:3, 3] = 1  # 3 fours
        game.dice[:, 3:, 4] = 1    # 2 fives
        
        # Game 0: score category 3 (valid)
        # Game 1: score category 4 (valid)
        category = torch.tensor([[3], [4]], dtype=torch.long)
        reward1 = game.score_category(category)
        
        self.assertEqual(reward1[0, 0].item(), 12, "Game 0: 3 fours * 4 = 12")
        self.assertEqual(reward1[1, 0].item(), 10, "Game 1: 2 fives * 5 = 10")
        
        # Now both try to score their already-scored categories
        reward2 = game.score_category(category)
        
        # The actual category scored will be random unscored one
        # The reward will be whatever that category scores, minus 10
        # Since dice have 3 fours and 2 fives, the reward depends on which category is chosen
        # But it should definitely be less than the original reward
        self.assertLess(reward2[0, 0].item(), reward1[0, 0].item(), 
                       "Penalized reward should be less than original")
        self.assertLess(reward2[1, 0].item(), reward1[1, 0].item(), 
                       "Penalized reward should be less than original")
        
        # And both should have penalties
        self.assertEqual(game.penalties[0, 0].item(), 1, "Game 0 should have penalty")
        self.assertEqual(game.penalties[1, 0].item(), 1, "Game 1 should have penalty")


if __name__ == '__main__':
    unittest.main()