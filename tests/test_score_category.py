import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from simple_yahtzee import SimpleYahtzee


class TestScoreCategory(unittest.TestCase):
    
    def test_score_category_updates_scores(self):
        """Test that score_category updates the scores tensor correctly."""
        game = SimpleYahtzee(2)
        
        # Set up specific dice: game 0 has three 2s, game 1 has two 4s
        game.dice = torch.zeros(2, 5, 6)
        # Game 0: three 2s (index 1), two 5s (index 4)
        game.dice[0, 0, 1] = 1
        game.dice[0, 1, 1] = 1
        game.dice[0, 2, 1] = 1
        game.dice[0, 3, 4] = 1
        game.dice[0, 4, 4] = 1
        
        # Game 1: two 4s (index 3), three 6s (index 5)
        game.dice[1, 0, 3] = 1
        game.dice[1, 1, 3] = 1
        game.dice[1, 2, 5] = 1
        game.dice[1, 3, 5] = 1
        game.dice[1, 4, 5] = 1
        
        # Score category 1 (2s) for game 0, category 3 (4s) for game 1
        category = torch.tensor([[1], [3]], dtype=torch.long)
        game.score_category(category)
        
        # Check game 0, category 1 should have count 3 (three 2s)
        # Pattern should be [0,0,0,1,0,0,0] (index 3 = count of 3)
        expected_game0_cat1 = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[0, 1], expected_game0_cat1))
        
        # Check game 1, category 3 should have count 2 (two 4s)
        # Pattern should be [0,0,1,0,0,0,0] (index 2 = count of 2)
        expected_game1_cat3 = torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[1, 3], expected_game1_cat3))
        
        # Other categories should remain unscored
        unscored_pattern = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        # Check game 0's other categories
        for cat in [0, 2, 3, 4, 5]:
            self.assertTrue(torch.equal(game.scores[0, cat], unscored_pattern))
        # Check game 1's other categories  
        for cat in [0, 1, 2, 4, 5]:
            self.assertTrue(torch.equal(game.scores[1, cat], unscored_pattern))
    
    def test_scoring_different_categories_different_games(self):
        """Test that scoring different categories in different games works correctly."""
        game = SimpleYahtzee(3)
        
        # Set up dice for all three games
        game.dice = torch.zeros(3, 5, 6)
        # All games have the same dice for simplicity: [1,1,2,3,4]
        for game_idx in range(3):
            game.dice[game_idx, 0, 0] = 1  # 1
            game.dice[game_idx, 1, 0] = 1  # 1
            game.dice[game_idx, 2, 1] = 1  # 2
            game.dice[game_idx, 3, 2] = 1  # 3
            game.dice[game_idx, 4, 3] = 1  # 4
        
        # Score different categories for each game
        # Game 0: category 0 (1s), Game 1: category 2 (3s), Game 2: category 3 (4s)
        category = torch.tensor([[0], [2], [3]], dtype=torch.long)
        game.score_category(category)
        
        # Check game 0, category 0 (1s) should have count 2
        expected_game0_cat0 = torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[0, 0], expected_game0_cat0))
        
        # Check game 1, category 2 (3s) should have count 1
        expected_game1_cat2 = torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[1, 2], expected_game1_cat2))
        
        # Check game 2, category 3 (4s) should have count 1
        expected_game2_cat3 = torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[2, 3], expected_game2_cat3))
    
    def test_scoring_all_possible_counts(self):
        """Test scoring values 0-5 for dice values 1-6 score correctly."""
        game = SimpleYahtzee(6)
        
        # Set up dice with counts 0-5 for category 0 (1s)
        game.dice = torch.zeros(6, 5, 6)
        
        # Game 0: 0 ones
        game.dice[0, :, 1:] = torch.tensor([[1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0]]).T
        
        # Game 1: 1 one
        game.dice[1, 0, 0] = 1
        game.dice[1, 1:, 1] = 1
        
        # Game 2: 2 ones
        game.dice[2, 0:2, 0] = 1
        game.dice[2, 2:, 1] = 1
        
        # Game 3: 3 ones
        game.dice[3, 0:3, 0] = 1
        game.dice[3, 3:, 1] = 1
        
        # Game 4: 4 ones
        game.dice[4, 0:4, 0] = 1
        game.dice[4, 4, 1] = 1
        
        # Game 5: 5 ones
        game.dice[5, :, 0] = 1
        
        # Score category 0 (1s) for all games
        category = torch.tensor([[0], [0], [0], [0], [0], [0]], dtype=torch.long)
        game.score_category(category)
        
        # Check each game has the correct count
        expected_patterns = [
            torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float32),  # 0 ones
            torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.float32),  # 1 one
            torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.float32),  # 2 ones
            torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32),  # 3 ones
            torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.float32),  # 4 ones
            torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype=torch.float32),  # 5 ones
        ]
        
        for game_idx, expected in enumerate(expected_patterns):
            self.assertTrue(torch.equal(game.scores[game_idx, 0], expected),
                           f"Game {game_idx} category 0 expected {expected} but got {game.scores[game_idx, 0]}")
    
    def test_unscored_categories_remain_unscored(self):
        """Test that categories that were unscored and not chosen remain unscored."""
        game = SimpleYahtzee(2)
        
        # Set up some dice
        game.dice = torch.zeros(2, 5, 6)
        game.dice[0, :, 0] = 1  # All 1s for game 0
        game.dice[1, :, 5] = 1  # All 6s for game 1
        
        # Score only category 0 for game 0 and category 5 for game 1
        category = torch.tensor([[0], [5]], dtype=torch.long)
        game.score_category(category)
        
        # Check that all other categories remain unscored
        unscored_pattern = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        
        # Game 0: categories 1-5 should be unscored
        for cat in range(1, 6):
            self.assertTrue(torch.equal(game.scores[0, cat], unscored_pattern),
                           f"Game 0 category {cat} should be unscored")
        
        # Game 1: categories 0-4 should be unscored
        for cat in range(0, 5):
            self.assertTrue(torch.equal(game.scores[1, cat], unscored_pattern),
                           f"Game 1 category {cat} should be unscored")
    
    def test_scored_categories_no_unscored_bit(self):
        """Test that categories that were scored don't have the 7th bit set."""
        game = SimpleYahtzee(3)
        
        # Set up dice
        game.dice = torch.zeros(3, 5, 6)
        # Different counts for each game
        game.dice[0, 0:2, 2] = 1  # 2 threes
        game.dice[0, 2:, 0] = 1   # Rest are ones
        
        game.dice[1, 0:4, 1] = 1  # 4 twos  
        game.dice[1, 4, 3] = 1     # 1 four
        
        game.dice[2, :, 4] = 1     # 5 fives
        
        # Score different categories
        category = torch.tensor([[2], [1], [4]], dtype=torch.long)
        game.score_category(category)
        
        # Check that scored categories have 7th bit (index 6) as 0
        self.assertEqual(game.scores[0, 2, 6].item(), 0, "Scored category should have 7th bit = 0")
        self.assertEqual(game.scores[1, 1, 6].item(), 0, "Scored category should have 7th bit = 0")
        self.assertEqual(game.scores[2, 4, 6].item(), 0, "Scored category should have 7th bit = 0")
        
        # Also verify they have the correct count bit set
        self.assertEqual(game.scores[0, 2, 2].item(), 1, "Should have count 2 for threes")
        self.assertEqual(game.scores[1, 1, 4].item(), 1, "Should have count 4 for twos")
        self.assertEqual(game.scores[2, 4, 5].item(), 1, "Should have count 5 for fives")
    
    def test_scoring_preserves_previous_scores(self):
        """Test that scoring a new category preserves previously scored categories."""
        game = SimpleYahtzee(1)
        
        # Score category 0 first
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, 0:3, 0] = 1  # 3 ones
        game.dice[0, 3:, 1] = 1    # 2 twos
        
        category = torch.tensor([[0]], dtype=torch.long)
        game.score_category(category)
        
        # Verify category 0 is scored with count 3
        expected_cat0 = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[0, 0], expected_cat0))
        
        # Now roll new dice and score category 1
        game.dice[0, :, :] = 0
        game.dice[0, 0:2, 1] = 1  # 2 twos
        game.dice[0, 2:, 2] = 1    # 3 threes
        
        category = torch.tensor([[1]], dtype=torch.long)
        game.score_category(category)
        
        # Verify category 0 still has its original score
        self.assertTrue(torch.equal(game.scores[0, 0], expected_cat0),
                       "Previous score should be preserved")
        
        # And category 1 has the new score
        expected_cat1 = torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[0, 1], expected_cat1))
    
    def test_scoring_zero_count(self):
        """Test that scoring when you have zero of a die value works correctly."""
        game = SimpleYahtzee(1)
        
        # Set dice with no 3s
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, :, 0] = 1  # All ones
        
        # Score category 2 (threes) 
        category = torch.tensor([[2]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Should have count 0 pattern: [1,0,0,0,0,0,0]
        expected = torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(game.scores[0, 2], expected),
                       f"Zero count should be [1,0,0,0,0,0,0] but got {game.scores[0, 2]}")
        
        # Reward should be 0 (0 threes * 3)
        self.assertEqual(reward[0, 0].item(), 0, "Reward for zero count should be 0")
    
    def test_reward_calculation_single_game(self):
        """Test that reward calculation is correct for a single game."""
        game = SimpleYahtzee(1)
        
        # Set dice: 3 fours and 2 twos
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, 0:3, 3] = 1  # 3 fours (index 3)
        game.dice[0, 3:, 1] = 1    # 2 twos (index 1)
        
        # Score category 3 (fours)
        category = torch.tensor([[3]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Reward should be 3 * 4 = 12
        self.assertEqual(reward.shape, (1, 1), "Reward should have shape (1, 1)")
        self.assertEqual(reward[0, 0].item(), 12, "Reward should be 3 fours * 4 = 12")
        
        # Now score category 1 (twos) with new dice
        game.dice[0, :, :] = 0
        game.dice[0, 0:4, 1] = 1  # 4 twos
        game.dice[0, 4, 5] = 1     # 1 six
        
        category = torch.tensor([[1]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Reward should be 4 * 2 = 8
        self.assertEqual(reward[0, 0].item(), 8, "Reward should be 4 twos * 2 = 8")
    
    def test_reward_calculation_multiple_games(self):
        """Test reward calculation for multiple parallel games."""
        game = SimpleYahtzee(4)
        
        # Set up different dice for each game
        game.dice = torch.zeros(4, 5, 6)
        
        # Game 0: 5 ones
        game.dice[0, :, 0] = 1
        
        # Game 1: 3 twos
        game.dice[1, 0:3, 1] = 1
        game.dice[1, 3:, 2] = 1  # Rest are threes
        
        # Game 2: 2 fives
        game.dice[2, 0:2, 4] = 1
        game.dice[2, 2:, 0] = 1  # Rest are ones
        
        # Game 3: 0 sixes
        game.dice[3, :, 3] = 1  # All fours
        
        # Score different categories
        category = torch.tensor([[0], [1], [4], [5]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Check rewards
        self.assertEqual(reward.shape, (4, 1), "Reward should have shape (4, 1)")
        self.assertEqual(reward[0, 0].item(), 5, "Game 0: 5 ones * 1 = 5")
        self.assertEqual(reward[1, 0].item(), 6, "Game 1: 3 twos * 2 = 6")
        self.assertEqual(reward[2, 0].item(), 10, "Game 2: 2 fives * 5 = 10")
        self.assertEqual(reward[3, 0].item(), 0, "Game 3: 0 sixes * 6 = 0")
    
    def test_reward_all_categories_all_counts(self):
        """Test rewards for all categories (1-6) with all possible counts (0-5)."""
        # 6 categories * 6 possible counts = 36 games
        game = SimpleYahtzee(36)
        
        # Set up dice and categories
        game.dice = torch.zeros(36, 5, 6)
        categories = []
        expected_rewards = []
        
        game_idx = 0
        for cat_idx in range(6):  # Categories 0-5 (ones through sixes)
            for count in range(6):  # Counts 0-5
                # Set up dice with 'count' of the target die
                if count > 0:
                    game.dice[game_idx, 0:count, cat_idx] = 1
                # Fill rest with something else
                if count < 5:
                    other_idx = (cat_idx + 1) % 6
                    game.dice[game_idx, count:, other_idx] = 1
                
                categories.append([cat_idx])
                expected_rewards.append(count * (cat_idx + 1))
                game_idx += 1
        
        # Score all categories
        category = torch.tensor(categories, dtype=torch.long)
        reward = game.score_category(category)
        
        # Check all rewards
        self.assertEqual(reward.shape, (36, 1), "Reward should have shape (36, 1)")
        for i, expected in enumerate(expected_rewards):
            self.assertEqual(reward[i, 0].item(), expected, 
                           f"Game {i}: expected reward {expected}, got {reward[i, 0].item()}")
    
    def test_reward_maximum_scores(self):
        """Test maximum possible rewards for each category."""
        game = SimpleYahtzee(6)
        
        # Set all dice to same value for each game
        game.dice = torch.zeros(6, 5, 6)
        for i in range(6):
            game.dice[i, :, i] = 1  # All dice show value i+1
        
        # Score each game with its matching category
        category = torch.tensor([[0], [1], [2], [3], [4], [5]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Check maximum rewards
        expected_max_rewards = [5, 10, 15, 20, 25, 30]  # 5*1, 5*2, 5*3, 5*4, 5*5, 5*6
        for i, expected in enumerate(expected_max_rewards):
            self.assertEqual(reward[i, 0].item(), expected,
                           f"Max reward for category {i} should be {expected}")


if __name__ == '__main__':
    unittest.main()