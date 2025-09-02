import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from simple_yahtzee import SimpleYahtzee
from game_runner import GameRunner
from random_player import RandomPlayer


class TestScoringConsistency(unittest.TestCase):
    
    def test_rewards_match_traditional_score(self):
        """Test that sum of rewards (without penalties) equals traditional Yahtzee score."""
        game = SimpleYahtzee(1)
        
        # Manually score several categories
        total_rewards = 0
        total_traditional_score = 0
        
        # Round 1: Score 3 ones
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, 0:3, 0] = 1  # 3 ones
        game.dice[0, 3:, 1] = 1    # 2 twos
        category = torch.tensor([[0]], dtype=torch.long)
        reward = game.score_category(category)
        total_rewards += reward[0, 0].item()
        total_traditional_score += 3 * 1  # 3 ones = 3 points
        
        # Round 2: Score 4 twos
        game.dice[:] = 0
        game.dice[0, 0:4, 1] = 1  # 4 twos
        game.dice[0, 4, 2] = 1     # 1 three
        category = torch.tensor([[1]], dtype=torch.long)
        reward = game.score_category(category)
        total_rewards += reward[0, 0].item()
        total_traditional_score += 4 * 2  # 4 twos = 8 points
        
        # Round 3: Score 2 fives
        game.dice[:] = 0
        game.dice[0, 0:2, 4] = 1  # 2 fives
        game.dice[0, 2:, 0] = 1    # 3 ones
        category = torch.tensor([[4]], dtype=torch.long)
        reward = game.score_category(category)
        total_rewards += reward[0, 0].item()
        total_traditional_score += 2 * 5  # 2 fives = 10 points
        
        self.assertEqual(total_rewards, total_traditional_score,
                        f"Rewards sum ({total_rewards}) should equal traditional score ({total_traditional_score})")
    
    def test_full_game_scoring_consistency(self):
        """Test that a full game's rewards match the traditional scoring."""
        player = RandomPlayer()
        runner = GameRunner(player)
        runner.create_game(5)
        
        # Play the game
        runner.play_game()
        trajectory = runner.get_trajectory()
        
        # Calculate total rewards (excluding penalties)
        rewards = trajectory['rewards']
        
        # Only category actions have non-zero rewards
        # These occur at indices 2, 5, 8, 11, 14, 17 (every 3rd action)
        category_rewards = [rewards[i] for i in range(2, len(rewards), 3)]
        
        # For each game, calculate traditional score from the scores tensor
        for game_idx in range(5):
            # Sum rewards for this game (before penalties)
            game_rewards = sum(r[game_idx, 0].item() for r in category_rewards)
            
            # Add back penalties that were subtracted
            penalties = runner.game.penalties[game_idx, 0].item()
            game_rewards_before_penalties = game_rewards + (penalties * 10)
            
            # Calculate traditional score from scores tensor
            traditional_score = 0
            for cat_idx in range(6):
                score_pattern = runner.game.scores[game_idx, cat_idx]
                if score_pattern[6] == 0:  # Category is scored (not unscored)
                    # Find the count
                    count = score_pattern[:6].nonzero()[0].item()
                    points = count * (cat_idx + 1)
                    traditional_score += points
            
            self.assertEqual(game_rewards_before_penalties, traditional_score,
                           f"Game {game_idx}: Rewards ({game_rewards_before_penalties}) != Traditional ({traditional_score})")
    
    def test_penalty_doesnt_affect_traditional_score(self):
        """Test that penalties affect rewards but not the traditional score."""
        game = SimpleYahtzee(1)
        
        # Score category 0 with 5 ones
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[0]], dtype=torch.long)
        reward1 = game.score_category(category)
        
        self.assertEqual(reward1[0, 0].item(), 5, "Should get 5 points for 5 ones")
        
        # Try to score same category again (penalty)
        game.dice[:] = 0
        game.dice[0, 0:3, 1] = 1  # 3 twos
        game.dice[0, 3:, 2] = 1    # 2 threes
        reward2 = game.score_category(category)
        
        # Reward should be negative due to penalty
        self.assertLess(reward2[0, 0].item(), 0, "Should get negative reward due to penalty")
        
        # But traditional score for category 0 should still be 5
        score_pattern = game.scores[0, 0]
        count = score_pattern[:6].nonzero()[0].item()
        traditional_score = count * 1
        self.assertEqual(traditional_score, 5, "Traditional score should still be 5 ones")
    
    def test_zero_scores_consistency(self):
        """Test that scoring 0 of a category works correctly."""
        game = SimpleYahtzee(1)
        
        # Score category 2 (threes) with no threes
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, :, 0] = 1  # All ones
        category = torch.tensor([[2]], dtype=torch.long)
        reward = game.score_category(category)
        
        # Reward should be 0
        self.assertEqual(reward[0, 0].item(), 0, "Should get 0 points for 0 threes")
        
        # Traditional score should also be 0
        score_pattern = game.scores[0, 2]
        count = score_pattern[:6].nonzero()[0].item()
        traditional_score = count * 3
        self.assertEqual(traditional_score, 0, "Traditional score should be 0")
    
    def test_maximum_scores_consistency(self):
        """Test that maximum scores are consistent."""
        game = SimpleYahtzee(1)
        
        total_rewards = 0
        total_traditional = 0
        
        # Score each category with maximum (5 of each)
        for cat_idx in range(6):
            game.dice = torch.zeros(1, 5, 6)
            game.dice[0, :, cat_idx] = 1  # All of this category
            category = torch.tensor([[cat_idx]], dtype=torch.long)
            reward = game.score_category(category)
            
            expected_score = 5 * (cat_idx + 1)
            self.assertEqual(reward[0, 0].item(), expected_score,
                           f"Category {cat_idx}: Should get {expected_score} points")
            
            total_rewards += reward[0, 0].item()
            total_traditional += expected_score
        
        # Maximum possible upper section score is 5+10+15+20+25+30 = 105
        self.assertEqual(total_rewards, 105, "Total rewards should be 105")
        self.assertEqual(total_traditional, 105, "Traditional score should be 105")


if __name__ == '__main__':
    unittest.main()