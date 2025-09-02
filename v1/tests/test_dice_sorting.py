import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import unittest
from simple_yahtzee import SimpleYahtzee


class TestDiceSorting(unittest.TestCase):
    
    def test_dice_sorted_after_roll(self):
        """Test that dice are sorted in ascending order after rolling."""
        game = SimpleYahtzee(5)
        game.roll_dice()
        
        # Convert one-hot to values
        dice_values = torch.argmax(game.dice, dim=2)
        
        # Check each game's dice are sorted
        for game_idx in range(5):
            dice = dice_values[game_idx]
            for i in range(4):
                self.assertLessEqual(dice[i].item(), dice[i+1].item(),
                                   f"Game {game_idx}: dice not sorted at position {i}")
    
    def test_dice_sorted_after_hold(self):
        """Test that dice remain sorted after applying hold mask."""
        game = SimpleYahtzee(1)
        
        # Set specific dice values (unsorted)
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, 0, 5] = 1  # 6
        game.dice[0, 1, 0] = 1  # 1
        game.dice[0, 2, 3] = 1  # 4
        game.dice[0, 3, 0] = 1  # 1
        game.dice[0, 4, 2] = 1  # 3
        # Current unsorted: [6, 1, 4, 1, 3]
        
        # After rolling with no hold, should be sorted
        game.roll_dice()
        dice_values = torch.argmax(game.dice, dim=2)
        
        # Verify sorted
        for i in range(4):
            self.assertLessEqual(dice_values[0, i].item(), dice_values[0, i+1].item())
    
    def test_hold_mask_applies_to_sorted_positions(self):
        """Test that hold mask applies to sorted dice positions and result is sorted."""
        game = SimpleYahtzee(1)
        
        # First roll
        game.roll_dice()
        initial_values = torch.argmax(game.dice, dim=2)
        
        # Get the two lowest values (first two in sorted order)
        held_values = [initial_values[0, 0].item(), initial_values[0, 1].item()]
        
        # Hold first two dice (in sorted order)
        hold_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32)
        
        # Roll again with hold
        game.roll_dice(hold_mask)
        new_values = torch.argmax(game.dice, dim=2)
        
        # The held values should appear somewhere in the new sorted dice
        new_values_list = new_values[0].tolist()
        for held_val in held_values:
            self.assertIn(held_val, new_values_list,
                         f"Held value {held_val} should be in new dice")
        
        # Result should still be sorted
        for i in range(4):
            self.assertLessEqual(new_values[0, i].item(), new_values[0, i+1].item(),
                               f"Dice not sorted at position {i}")
    
    def test_multiple_holds_maintain_sorting(self):
        """Test that multiple hold/roll cycles maintain sorting."""
        game = SimpleYahtzee(2)
        
        # Roll and hold several times
        for _ in range(5):
            game.roll_dice()
            
            # Random hold pattern
            hold_mask = torch.randint(0, 2, (2, 5), dtype=torch.float32)
            game.roll_dice(hold_mask)
            
            # Check still sorted
            dice_values = torch.argmax(game.dice, dim=2)
            for game_idx in range(2):
                for i in range(4):
                    self.assertLessEqual(dice_values[game_idx, i].item(), 
                                       dice_values[game_idx, i+1].item(),
                                       f"Game {game_idx}: dice not sorted")
    
    def test_identical_dice_sorting(self):
        """Test that identical dice values are handled correctly."""
        game = SimpleYahtzee(1)
        
        # Manually set all dice to same value
        game.dice = torch.zeros(1, 5, 6)
        game.dice[0, :, 2] = 1  # All threes
        
        # Apply hold mask and roll
        hold_mask = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.float32)
        game.roll_dice(hold_mask)
        
        # Should still be sorted
        dice_values = torch.argmax(game.dice, dim=2)
        for i in range(4):
            self.assertLessEqual(dice_values[0, i].item(), dice_values[0, i+1].item())
    
    def test_sorting_preserves_one_hot(self):
        """Test that sorting preserves valid one-hot encoding."""
        game = SimpleYahtzee(10)
        game.roll_dice()
        
        # Each die should still be one-hot encoded
        sums = game.dice.sum(dim=2)
        self.assertTrue(torch.all(sums == 1), "Each die should have exactly one 1")
        
        # All values should be 0 or 1
        self.assertTrue(torch.all((game.dice == 0) | (game.dice == 1)),
                       "All values should be 0 or 1")


if __name__ == '__main__':
    unittest.main()