import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from yahtzee import Yahtzee


class TestDiceRolling(unittest.TestCase):
    
    def test_initial_dice_shape(self):
        """Test that dice tensor has correct shape on initialization."""
        game = Yahtzee(10)
        self.assertEqual(game.dice.shape, (10, 5, 6))
        
    def test_initial_dice_zeros(self):
        """Test that dice tensor starts with all zeros."""
        game = Yahtzee(5)
        self.assertTrue(torch.all(game.dice == 0))
        
    def test_roll_all_dice_shape(self):
        """Test that rolling all dice maintains correct shape."""
        game = Yahtzee(8)
        game.roll_dice()
        self.assertEqual(game.dice.shape, (8, 5, 6))
        
    def test_roll_all_dice_one_hot(self):
        """Test that rolled dice are valid one-hot encoded."""
        game = Yahtzee(10)
        game.roll_dice()
        
        # Each die should have exactly one 1 and five 0s
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1))
        
        # All values should be 0 or 1
        self.assertTrue(torch.all((game.dice == 0) | (game.dice == 1)))
        
    def test_roll_dice_randomness(self):
        """Test that dice rolls produce different results."""
        game = Yahtzee(100)
        game.roll_dice()
        first_roll = game.dice.clone()
        
        game.roll_dice()
        second_roll = game.dice.clone()
        
        # Should not be identical (extremely unlikely with 500 dice)
        self.assertFalse(torch.equal(first_roll, second_roll))
        
    def test_hold_all_dice(self):
        """Test that holding all dice keeps them unchanged."""
        game = Yahtzee(5)
        game.roll_dice()
        initial_dice = game.dice.clone()
        
        # Hold all dice (all 1s)
        hold_mask = torch.ones(5, 5, dtype=torch.bool, device=game.device)
        game.roll_dice(hold_mask)
        
        # Dice should be unchanged
        self.assertTrue(torch.equal(initial_dice, game.dice))
        
    def test_hold_no_dice(self):
        """Test that holding no dice rerolls all dice."""
        game = Yahtzee(5)
        game.roll_dice()
        initial_dice = game.dice.clone()
        
        # Hold no dice (all 0s)
        hold_mask = torch.zeros(5, 5, dtype=torch.bool, device=game.device)
        game.roll_dice(hold_mask)
        
        # Still valid one-hot
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1))
        
        # Should be different (very likely)
        self.assertFalse(torch.equal(initial_dice, game.dice))
        
    def test_hold_some_dice(self):
        """Test that holding specific dice works correctly."""
        game = Yahtzee(3)
        game.roll_dice()
        initial_dice = game.dice.clone()
        
        # Hold dice 0, 2, 4 for all games
        hold_mask = torch.tensor([
            [True, False, True, False, True],
            [True, False, True, False, True],
            [True, False, True, False, True]
        ], dtype=torch.bool, device=game.device)
        
        game.roll_dice(hold_mask)
        
        # Check that the held dice values appear in the result (may be in different positions due to sorting)
        initial_values = torch.argmax(initial_dice, dim=2)
        final_values = torch.argmax(game.dice, dim=2)
        
        # The held dice values should be present in the final dice
        for game_idx in range(3):
            held_vals = [initial_values[game_idx, 0].item(), 
                        initial_values[game_idx, 2].item(), 
                        initial_values[game_idx, 4].item()]
            final_vals = final_values[game_idx].tolist()
            for held_val in held_vals:
                self.assertIn(held_val, final_vals, 
                             f"Held value {held_val} should be in final dice for game {game_idx}")
        
        # Unheld dice should still be valid one-hot
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1))
        
    def test_clear_resets_dice(self):
        """Test that clear() resets dice to zeros."""
        game = Yahtzee(5)
        game.roll_dice()
        
        # Verify dice are not zero
        self.assertFalse(torch.all(game.dice == 0))
        
        game.clear()
        
        # Verify dice are now zero
        self.assertTrue(torch.all(game.dice == 0))
        
    def test_different_game_different_rolls(self):
        """Test that different parallel games get different rolls."""
        game = Yahtzee(10)
        game.roll_dice()
        
        # Check that not all games have identical dice
        # (extremely unlikely for all 10 games to roll exactly the same)
        first_game = game.dice[0]
        all_same = True
        for i in range(1, 10):
            if not torch.equal(first_game, game.dice[i]):
                all_same = False
                break
                
        self.assertFalse(all_same)
    
    def test_initial_upper_scores_unscored(self):
        """Test that upper scores tensor initializes with unscored pattern."""
        game = Yahtzee(5)
        
        # Check shape
        self.assertEqual(game.upper_scores.shape, (5, 6, 7))
        
        # Check that all categories are marked as unscored [0,0,0,0,0,0,1]
        expected_pattern = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device=game.device)
        for game_idx in range(5):
            for category_idx in range(6):
                self.assertTrue(torch.equal(game.upper_scores[game_idx, category_idx], expected_pattern))
    
    def test_initial_lower_scores_unscored(self):
        """Test that lower scores tensor initializes correctly."""
        game = Yahtzee(5)
        
        # Check shape
        self.assertEqual(game.lower_scores.shape, (5, 7, 2))
        
        # Check that all categories are marked as unscored [0, 1]
        for game_idx in range(5):
            for category_idx in range(7):
                self.assertEqual(game.lower_scores[game_idx, category_idx, 0].item(), 0)  # points = 0
                self.assertEqual(game.lower_scores[game_idx, category_idx, 1].item(), 1)  # unscored = 1
    
    def test_clear_resets_upper_scores_to_unscored(self):
        """Test that clear() resets upper scores to unscored pattern."""
        game = Yahtzee(3)
        
        # Modify scores tensor
        game.upper_scores[:, :, :] = 0
        game.upper_scores[:, :, 2] = 1  # Set some arbitrary pattern
        
        # Clear should reset to unscored
        game.clear()
        
        # Check that all categories are marked as unscored [0,0,0,0,0,0,1]
        expected_pattern = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device=game.device)
        for game_idx in range(3):
            for category_idx in range(6):
                self.assertTrue(torch.equal(game.upper_scores[game_idx, category_idx], expected_pattern))
    
    def test_clear_resets_lower_scores_to_unscored(self):
        """Test that clear() resets lower scores to unscored pattern."""
        game = Yahtzee(3)
        
        # Modify scores tensor
        game.lower_scores[:, :, 0] = 10  # Set some points
        game.lower_scores[:, :, 1] = 0   # Mark as scored
        
        # Clear should reset to unscored
        game.clear()
        
        # Check that all categories are reset
        for game_idx in range(3):
            for category_idx in range(7):
                self.assertEqual(game.lower_scores[game_idx, category_idx, 0].item(), 0)  # points = 0
                self.assertEqual(game.lower_scores[game_idx, category_idx, 1].item(), 1)  # unscored = 1


if __name__ == '__main__':
    unittest.main()