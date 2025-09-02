import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from simple_yahtzee import SimpleYahtzee


class TestDiceRolling(unittest.TestCase):
    
    def test_initial_dice_shape(self):
        """Test that dice tensor has correct shape on initialization."""
        game = SimpleYahtzee(10)
        self.assertEqual(game.dice.shape, (10, 5, 6))
        
    def test_initial_dice_zeros(self):
        """Test that dice tensor starts with all zeros."""
        game = SimpleYahtzee(5)
        self.assertTrue(torch.all(game.dice == 0))
        
    def test_roll_all_dice_shape(self):
        """Test that rolling all dice maintains correct shape."""
        game = SimpleYahtzee(8)
        game.roll_dice()
        self.assertEqual(game.dice.shape, (8, 5, 6))
        
    def test_roll_all_dice_one_hot(self):
        """Test that rolled dice are valid one-hot encoded."""
        game = SimpleYahtzee(10)
        game.roll_dice()
        
        # Each die should have exactly one 1 and five 0s
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1))
        
        # All values should be 0 or 1
        self.assertTrue(torch.all((game.dice == 0) | (game.dice == 1)))
        
    def test_roll_dice_randomness(self):
        """Test that dice rolls produce different results."""
        game = SimpleYahtzee(100)
        game.roll_dice()
        first_roll = game.dice.clone()
        
        game.roll_dice()
        second_roll = game.dice.clone()
        
        # Should not be identical (extremely unlikely with 500 dice)
        self.assertFalse(torch.equal(first_roll, second_roll))
        
    def test_hold_all_dice(self):
        """Test that holding all dice keeps them unchanged."""
        game = SimpleYahtzee(5)
        game.roll_dice()
        initial_dice = game.dice.clone()
        
        # Hold all dice (all 1s)
        hold_mask = torch.ones(5, 5)
        game.roll_dice(hold_mask)
        
        # Dice should be unchanged
        self.assertTrue(torch.equal(initial_dice, game.dice))
        
    def test_hold_no_dice(self):
        """Test that holding no dice rerolls all dice."""
        game = SimpleYahtzee(5)
        game.roll_dice()
        initial_dice = game.dice.clone()
        
        # Hold no dice (all 0s)
        hold_mask = torch.zeros(5, 5)
        game.roll_dice(hold_mask)
        
        # Still valid one-hot
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1))
        
        # Should be different (very likely)
        self.assertFalse(torch.equal(initial_dice, game.dice))
        
    def test_hold_some_dice(self):
        """Test that holding specific dice works correctly."""
        game = SimpleYahtzee(3)
        game.roll_dice()
        initial_dice = game.dice.clone()
        
        # Hold dice 0, 2, 4 for all games
        hold_mask = torch.tensor([
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1]
        ], dtype=torch.float32)
        
        game.roll_dice(hold_mask)
        
        # Check held dice are unchanged
        self.assertTrue(torch.equal(initial_dice[:, 0, :], game.dice[:, 0, :]))
        self.assertTrue(torch.equal(initial_dice[:, 2, :], game.dice[:, 2, :]))
        self.assertTrue(torch.equal(initial_dice[:, 4, :], game.dice[:, 4, :]))
        
        # Unheld dice should still be valid one-hot
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1))
        
    def test_clear_resets_dice(self):
        """Test that clear() resets dice to zeros."""
        game = SimpleYahtzee(5)
        game.roll_dice()
        
        # Verify dice are not zero
        self.assertFalse(torch.all(game.dice == 0))
        
        game.clear()
        
        # Verify dice are now zero
        self.assertTrue(torch.all(game.dice == 0))
        
    def test_different_game_different_rolls(self):
        """Test that different parallel games get different rolls."""
        game = SimpleYahtzee(10)
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
    
    def test_initial_scores_unscored(self):
        """Test that scores tensor initializes with unscored pattern."""
        game = SimpleYahtzee(5)
        
        # Check shape
        self.assertEqual(game.scores.shape, (5, 6, 7))
        
        # Check that all categories are marked as unscored [0,0,0,0,0,0,1]
        expected_pattern = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        for game_idx in range(5):
            for category_idx in range(6):
                self.assertTrue(torch.equal(game.scores[game_idx, category_idx], expected_pattern))
    
    def test_clear_resets_scores_to_unscored(self):
        """Test that clear() resets scores to unscored pattern."""
        game = SimpleYahtzee(3)
        
        # Modify scores tensor
        game.scores[:, :, :] = 0
        game.scores[:, :, 2] = 1  # Set some arbitrary pattern
        
        # Clear should reset to unscored
        game.clear()
        
        # Check that all categories are marked as unscored [0,0,0,0,0,0,1]
        expected_pattern = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
        for game_idx in range(3):
            for category_idx in range(6):
                self.assertTrue(torch.equal(game.scores[game_idx, category_idx], expected_pattern))


if __name__ == '__main__':
    unittest.main()