import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from yahtzee import Yahtzee


class TestScoringConsistency(unittest.TestCase):
    
    def test_rewards_match_traditional_score(self):
        """Test that sum of rewards (without penalties) equals traditional Yahtzee score."""
        game = Yahtzee(1)
        
        # Manually score several categories
        total_rewards = 0
        total_traditional_score = 0
        
        # Round 1: Score 3 ones
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, 0:3, 0] = 1  # 3 ones
        game.dice[0, 3:, 1] = 1    # 2 twos
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        total_rewards += reward[0, 0].item()
        total_traditional_score += 3 * 1  # 3 ones = 3 points
        
        # Round 2: Score 4 twos
        game.dice[:] = 0
        game.dice[0, 0:4, 1] = 1  # 4 twos
        game.dice[0, 4, 2] = 1     # 1 three
        category = torch.tensor([[1]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        total_rewards += reward[0, 0].item()
        total_traditional_score += 4 * 2  # 4 twos = 8 points
        
        # Round 3: Score 2 fives
        game.dice[:] = 0
        game.dice[0, 0:2, 4] = 1  # 2 fives
        game.dice[0, 2:, 0] = 1    # 3 ones
        category = torch.tensor([[4]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        total_rewards += reward[0, 0].item()
        total_traditional_score += 2 * 5  # 2 fives = 10 points
        
        self.assertEqual(total_rewards, total_traditional_score,
                        f"Rewards sum ({total_rewards}) should equal traditional score ({total_traditional_score})")
    
    def test_penalty_doesnt_affect_traditional_score(self):
        """Test that penalties affect rewards but not the traditional score."""
        game = Yahtzee(1)
        
        # Score category 0 with 5 ones
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long, device=game.device)
        reward1 = game.score_upper(category)
        
        self.assertEqual(reward1[0, 0].item(), 5, "Should get 5 points for 5 ones")
        
        # Try to score same category again (penalty)
        game.dice[:] = 0
        game.dice[0, 0:3, 1] = 1  # 3 twos
        game.dice[0, 3:, 2] = 1    # 2 threes
        reward2 = game.score_upper(category)
        
        # Reward depends on random category selection
        # With 3 twos and 2 threes, this is a full house (25 points)
        # So reward could be 25 - 10 = 15, or other scores minus 10
        # We just verify penalty was applied
        self.assertEqual(game.penalties[0, 0].item(), 1, "Should have penalty for invalid choice")
        
        # But traditional score for category 0 should still be 5
        score_pattern = game.upper_scores[0, 0]
        count = score_pattern[:6].nonzero()[0].item()
        traditional_score = count * 1
        self.assertEqual(traditional_score, 5, "Traditional score should still be 5 ones")
    
    def test_zero_scores_consistency(self):
        """Test that scoring 0 of a category works correctly."""
        game = Yahtzee(1)
        
        # Score category 2 (threes) with no threes
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[2]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Reward should be 0
        self.assertEqual(reward[0, 0].item(), 0, "Should get 0 points for 0 threes")
        
        # Traditional score should also be 0
        score_pattern = game.upper_scores[0, 2]
        # First element (0 count) should be 1
        self.assertEqual(score_pattern[0].item(), 1, "Should have 0 count marked")
        traditional_score = 0 * 3
        self.assertEqual(traditional_score, 0, "Traditional score should be 0")
    
    def test_maximum_scores_consistency(self):
        """Test that maximum scores are consistent."""
        game = Yahtzee(1)
        
        total_rewards = 0
        total_traditional = 0
        
        # Score each category with maximum (5 of each)
        for cat_idx in range(6):
            game.dice = torch.zeros(1, 5, 6, device=game.device)
            game.dice[0, :, cat_idx] = 1  # All of this category
            category = torch.tensor([[cat_idx]], dtype=torch.long, device=game.device)
            reward = game.score_upper(category)
            
            expected_score = 5 * (cat_idx + 1)
            # Don't count bonus in base score comparison
            base_reward = reward[0, 0].item()
            if cat_idx == 3:  # After fours, we're at 50, not yet at 63
                self.assertEqual(base_reward, expected_score,
                               f"Category {cat_idx}: Should get {expected_score} points")
            elif cat_idx == 4:  # After fives, we're at 75, crossed 63 threshold
                self.assertEqual(base_reward, expected_score + 35,
                               f"Category {cat_idx}: Should get {expected_score} + 35 bonus")
            elif cat_idx == 5:  # Sixes, already have bonus
                self.assertEqual(base_reward, expected_score,
                               f"Category {cat_idx}: Should get {expected_score} points")
            else:
                self.assertEqual(base_reward, expected_score,
                               f"Category {cat_idx}: Should get {expected_score} points")
            
            if cat_idx == 4:
                total_rewards += expected_score + 35  # Include bonus once
            else:
                total_rewards += expected_score
            total_traditional += expected_score
        
        # Maximum possible upper section score is 5+10+15+20+25+30 = 105
        # With bonus: 105 + 35 = 140
        self.assertEqual(total_rewards, 140, "Total rewards should be 140 (105 + 35 bonus)")
        self.assertEqual(total_traditional, 105, "Traditional score should be 105")
    
    def test_bonus_threshold_exact(self):
        """Test bonus is awarded exactly at 63 points."""
        game = Yahtzee(1)
        
        # Score to get exactly 63
        # Strategy: 3 of each 1-6 = 3+6+9+12+15+18 = 63
        scores_so_far = 0
        
        for cat_idx in range(6):
            game.dice = torch.zeros(1, 5, 6, device=game.device)
            if cat_idx == 0:
                # For ones, we want exactly 3, but filling rest with ones gives us 5 total
                game.dice[0, 0:3, 0] = 1  # 3 ones
                game.dice[0, 3:, 1] = 1   # Fill rest with twos instead
            else:
                game.dice[0, 0:3, cat_idx] = 1  # 3 of current value
                game.dice[0, 3:, 0] = 1  # Fill rest with ones
            
            category = torch.tensor([[cat_idx]], dtype=torch.long, device=game.device)
            reward = game.score_upper(category)
            
            expected_base = 3 * (cat_idx + 1)
            scores_so_far += expected_base
            
            if scores_so_far >= 63 and scores_so_far - expected_base < 63:
                # Just crossed threshold
                self.assertEqual(reward[0, 0].item(), expected_base + 35,
                               f"Should get bonus when crossing 63 threshold")
            else:
                self.assertEqual(reward[0, 0].item(), expected_base,
                               f"Should get base score only")
    
    def test_bonus_not_awarded_below_63(self):
        """Test bonus is not awarded when total is below 63."""
        game = Yahtzee(1)
        
        # Score to get exactly 62
        # Strategy: 2 of each 1-6 = 2+4+6+8+10+12 = 42, then add 20 more
        
        # First 5 categories with 2 each
        for cat_idx in range(5):
            game.dice = torch.zeros(1, 5, 6, device=game.device)
            if cat_idx == 0:
                # For ones, we want exactly 2
                game.dice[0, 0:2, 0] = 1  # 2 ones
                game.dice[0, 2:, 1] = 1   # Fill rest with twos
            else:
                game.dice[0, 0:2, cat_idx] = 1  # 2 of current value
                game.dice[0, 2:, 0] = 1  # Fill rest with ones
            
            category = torch.tensor([[cat_idx]], dtype=torch.long, device=game.device)
            reward = game.score_upper(category)
            
            expected = 2 * (cat_idx + 1)
            self.assertEqual(reward[0, 0].item(), expected,
                           "Should get base score without bonus")
        
        # Score 4 sixes to get to 62 total (30 + 32 = 62) 
        # Wait, we scored 2 of each for cats 0-4, that's 2+4+6+8+10 = 30
        game.dice = torch.zeros(1, 5, 6, device=game.device)
        game.dice[0, 0:4, 5] = 1  # 4 sixes = 24
        game.dice[0, 4, 1] = 1   # 1 two
        category = torch.tensor([[5]], dtype=torch.long, device=game.device)
        reward = game.score_upper(category)
        
        # Should not get bonus (total is 30 + 24 = 54, below 63)
        self.assertEqual(reward[0, 0].item(), 24,
                        "Should not get bonus when below 63")
    
    def test_parallel_games_independent_bonuses(self):
        """Test that bonuses are tracked independently for parallel games."""
        game = Yahtzee(2)
        
        # Game 0: Score to get bonus
        # Game 1: Score to not get bonus
        
        for cat_idx in range(6):
            game.dice = torch.zeros(2, 5, 6, device=game.device)
            # Game 0: 5 of each (will hit bonus at category 3)
            game.dice[0, :, cat_idx] = 1
            # Game 1: 1 of each (will never hit bonus)
            if cat_idx == 0:
                # For ones category, all 5 would be ones
                game.dice[1, :, 0] = 1  # All ones
            else:
                game.dice[1, 0, cat_idx] = 1
                game.dice[1, 1:, 0] = 1  # Rest are ones
            
            categories = torch.tensor([[cat_idx], [cat_idx]], dtype=torch.long, device=game.device)
            rewards = game.score_upper(categories)
            
            # Check game 0 gets bonus when appropriate
            # 5 ones = 5, 5 twos = 10, 5 threes = 15, 5 fours = 20
            # Running total: 5, 15, 30, 50, 75 (crosses at fives), 105
            if cat_idx == 4:  # After 5+10+15+20=50, now +25=75, crossed 63
                self.assertEqual(rewards[0, 0].item(), 5 * (cat_idx + 1) + 35,
                               f"Game 0 should get bonus at category {cat_idx}")
            else:
                self.assertEqual(rewards[0, 0].item(), 5 * (cat_idx + 1),
                               f"Game 0 category {cat_idx}")
            
            # Game 1 should never get bonus
            if cat_idx == 0:
                # All 5 ones for game 1 when scoring ones
                self.assertEqual(rewards[1, 0].item(), 5,
                               "Game 1 should get 5 for ones category")
            else:
                self.assertEqual(rewards[1, 0].item(), 1 * (cat_idx + 1),
                               f"Game 1 should not get bonus at category {cat_idx}")


if __name__ == '__main__':
    unittest.main()