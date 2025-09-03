import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from yahtzee import Yahtzee


class TestScoreCategory(unittest.TestCase):
    
    def test_score_ones(self):
        """Test scoring ones category."""
        game = Yahtzee(1)
        
        # Set dice to have 3 ones and 2 twos
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, 0:3, 0] = 1  # 3 ones
        game.dice[0, 3:, 1] = 1    # 2 twos
        
        # Score the ones category
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should get 3 points (3 ones * 1)
        self.assertEqual(reward[0, 0].item(), 3)
        
        # Check that scores tensor is updated correctly
        # Should have [0,0,0,1,0,0,0] for 3 ones
        expected_score = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32, device=game.device)
        self.assertTrue(torch.equal(game.upper_scores[0, 0], expected_score))
    
    def test_score_sixes(self):
        """Test scoring sixes category."""
        game = Yahtzee(1)
        
        # Set dice to have 4 sixes
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, 0:4, 5] = 1  # 4 sixes
        game.dice[0, 4, 0] = 1     # 1 one
        
        # Score the sixes category
        category = torch.tensor([[5]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should get 24 points (4 sixes * 6)
        self.assertEqual(reward[0, 0].item(), 24)
        
        # Check that scores tensor is updated correctly
        # Should have [0,0,0,0,1,0,0] for 4 sixes
        expected_score = torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.float32, device=game.device)
        self.assertTrue(torch.equal(game.upper_scores[0, 5], expected_score))
    
    def test_score_zero_count(self):
        """Test scoring a category with zero matching dice."""
        game = Yahtzee(1)
        
        # Set dice to have no threes
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 0] = 1  # All ones
        
        # Score the threes category
        category = torch.tensor([[2]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should get 0 points (0 threes * 3)
        self.assertEqual(reward[0, 0].item(), 0)
        
        # Check that scores tensor is updated correctly
        # Should have [1,0,0,0,0,0,0] for 0 threes
        expected_score = torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device=game.device)
        self.assertTrue(torch.equal(game.upper_scores[0, 2], expected_score))
    
    def test_score_yahtzee_ones(self):
        """Test scoring all five of a kind in ones."""
        game = Yahtzee(1)
        
        # Set dice to have 5 ones (Yahtzee of ones)
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 0] = 1  # All ones
        
        # Score the ones category
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should get 5 points (5 ones * 1)
        self.assertEqual(reward[0, 0].item(), 5)
        
        # Check that scores tensor is updated correctly
        # Should have [0,0,0,0,0,1,0] for 5 ones
        expected_score = torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype=torch.float32, device=game.device)
        self.assertTrue(torch.equal(game.upper_scores[0, 0], expected_score))
    
    def test_parallel_games_different_categories(self):
        """Test scoring different categories in parallel games."""
        game = Yahtzee(3)
        
        # Set same dice for all games
        game.dice = torch.zeros(3, 5, 6, device=game.device)
        game.dice[:, 0:2, 1] = 1  # 2 twos
        game.dice[:, 2:, 2] = 1    # 3 threes
        
        # Score different categories for each game
        categories = torch.tensor([[1], [2], [1]], dtype=torch.long, device=game.device)
        rewards = game.score_upper(categories)
        
        # Game 0: twos -> 2*2 = 4
        # Game 1: threes -> 3*3 = 9
        # Game 2: twos -> 2*2 = 4
        self.assertEqual(rewards[0, 0].item(), 4)
        self.assertEqual(rewards[1, 0].item(), 9)
        self.assertEqual(rewards[2, 0].item(), 4)
    
    def test_score_already_scored_category(self):
        """Test that scoring an already scored category triggers penalty logic."""
        game = Yahtzee(1)
        
        # First, score a category
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 3] = 1  # All fours
        
        category = torch.tensor([[3]], dtype=torch.long, device=game.device)
        reward1 = game.score_upper(category)
        self.assertEqual(reward1[0, 0].item(), 20)  # 5 fours * 4
        
        # Try to score the same category again
        game.dice[:] = 0
        game.dice[0, :, 0] = 1  # All ones
        reward2 = game.score_upper(category)
        
        # Should trigger penalty - reward should be negative or much lower
        # The actual category scored will be random, but penalty of -10 applied
        self.assertLess(reward2[0, 0].item(), reward1[0, 0].item())
        
        # Original category should still show its original score
        # Should have [0,0,0,0,0,1,0] for 5 fours
        expected_score = torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype=torch.float32, device=game.device)
        self.assertTrue(torch.equal(game.upper_scores[0, 3], expected_score))
    
    def test_categories_marked_as_scored(self):
        """Test that categories are properly marked as scored after scoring."""
        game = Yahtzee(1)
        
        # Initially all categories should be unscored (last bit = 1)
        self.assertTrue(torch.all(game.upper_scores[0, :, 6] == 1))
        
        # Score category 2
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, 0:2, 2] = 1  # 2 threes
        game.dice[0, 2:, 1] = 1    # 3 twos
        
        category = torch.tensor([[2]], dtype=torch.long, device=game.device)
        game.score_upper(category)
        
        # Category 2 should now be scored (last bit = 0)
        self.assertEqual(game.upper_scores[0, 2, 6].item(), 0)
        
        # Others should still be unscored
        for i in [0, 1, 3, 4, 5]:
            self.assertEqual(game.upper_scores[0, i, 6].item(), 1)
    
    def test_score_lower_category_ignored(self):
        """Test that lower section categories are ignored in score_upper."""
        game = Yahtzee(1)
        
        # Set dice
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 2] = 1  # All threes
        
        # Try to score a lower section category (e.g., 8 = small straight)
        category = torch.tensor([[8]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should return 0 (no upper section scoring happened)
        self.assertEqual(reward[0, 0].item(), 0)
        
        # All upper categories should remain unscored
        self.assertTrue(torch.all(game.upper_scores[0, :, 6] == 1))
    
    def test_upper_bonus_calculation(self):
        """Test that upper section bonus is calculated correctly."""
        game = Yahtzee(1)
        
        # Score categories to reach exactly 63
        # 3 ones = 3, 3 twos = 6, 3 threes = 9, 3 fours = 12, 3 fives = 15, 3 sixes = 18
        # Total = 3 + 6 + 9 + 12 + 15 + 18 = 63
        
        for cat in range(6):
            game.dice = torch.zeros(1, 5, 6, device=game.device)
            if cat == 0:
                # For ones category, we actually have 5 dice (2 ones + 3 of cat 0 = 5 ones total) 
                game.dice[0, :, cat] = 1  # All 5 ones
            else:
                game.dice[0, 0:3, cat] = 1  # 3 of current category
                game.dice[0, 3:, 0] = 1     # 2 ones
            
            category = torch.tensor([[cat]], dtype=torch.long, device=game.device)
            reward = game.score_upper(category)
            
            # Check base reward is correct
            if cat == 0:
                expected = 5  # 5 ones
            else:
                expected = 3 * (cat + 1)  # 3 of each other category
            
            if cat == 5:  # Last category crosses the threshold
                # Should get 18 (3 sixes * 6) + 35 (bonus)
                self.assertEqual(reward[0, 0].item(), 18 + 35)
            else:
                # Should get normal score
                self.assertEqual(reward[0, 0].item(), expected)
    
    def test_upper_bonus_only_once(self):
        """Test that upper bonus is only awarded once."""
        game = Yahtzee(1)
        
        # First get to 63+ by scoring all categories with max
        for cat in range(6):
            game.dice = torch.zeros(1, 5, 6, device=game.device)
            game.dice[0, :, cat] = 1  # All of that value
            
            category = torch.tensor([[cat]], dtype=torch.long, device=game.device)
            reward = game.score_upper(category)
        
        # At this point we should have the bonus (total = 105, well over 63)
        # Try to score an already-scored category (will pick random unscored, but all are scored)
        game.dice[:] = 0
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        
        # This should trigger penalty logic, and bonus difference should be 0
        reward = game.score_upper(category)
        
        # Reward should be negative (penalty applied) and no bonus added
        self.assertLess(reward[0, 0].item(), 0)


if __name__ == '__main__':
    unittest.main()