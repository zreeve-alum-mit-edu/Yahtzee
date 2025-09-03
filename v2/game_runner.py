from yahtzee import Yahtzee
import torch
from device_config import device


class GameRunner:
    def __init__(self, player):
        """
        Initialize GameRunner with a player object.
        
        Args:
            player: Player object that will make decisions during the game
        """
        self.player = player
        # SAR tracking for PPO training
        self.states = []
        self.actions = []
        self.rewards = []
    
    def create_game(self, num_games):
        """
        Create a new Yahtzee game with specified number of parallel games.
        
        Args:
            num_games: Number of games to play in parallel
        """
        self.game = Yahtzee(num_games)
        # Clear SAR history when creating new game
        self.states = []
        self.actions = []
        self.rewards = []
    
    def get_state(self):
        """
        Get current game state from the player.
        
        Returns:
            Tensor containing game state
        """
        return self.player.get_state_from_game(self.game)
    
    def play_game(self):
        """
        Play a complete game of full Yahtzee (13 rounds).
        Tracks states, actions, and rewards for PPO training.
        """
        # Clear the game state and SAR history
        self.game.clear()
        self.states = []
        self.actions = []
        self.rewards = []
        
        # Play 13 rounds (one for each category: 6 upper + 7 lower)
        for round_num in range(13):
            # Update the round tracker
            self.game.round[:] = round_num
            # First roll - roll all dice
            self.game.roll_dice()
            
            # Set turn to first roll
            self.game.turn.zero_()
            self.game.turn[:, 0] = 1
            
            # First hold decision
            state = self.get_state()
            hold_mask = self.player.decide_hold(self.game)
            self.states.append(state)
            self.actions.append(('hold', hold_mask))
            # No immediate reward for hold decisions
            self.rewards.append(torch.zeros(self.game.num_games, 1, device=device))
            
            # Second roll - roll unheld dice
            self.game.roll_dice(hold_mask)
            
            # Set turn to second roll
            self.game.turn.zero_()
            self.game.turn[:, 1] = 1
            
            # Second hold decision
            state = self.get_state()
            hold_mask = self.player.decide_hold(self.game)
            self.states.append(state)
            self.actions.append(('hold', hold_mask))
            # No immediate reward for hold decisions
            self.rewards.append(torch.zeros(self.game.num_games, 1, device=device))
            
            # Third roll - roll unheld dice
            self.game.roll_dice(hold_mask)
            
            # Set turn to third roll (final)
            self.game.turn.zero_()
            self.game.turn[:, 2] = 1
            
            # Scoring decision
            state = self.get_state()
            category = self.player.decide_category(self.game)
            self.states.append(state)
            self.actions.append(('category', category))
            
            # Score the selected category and get reward
            reward = self.game.score_category(category)
            self.rewards.append(reward)
    
    def get_trajectory(self):
        """
        Get the complete trajectory of the last game played.
        
        Returns:
            Dictionary containing:
                - states: List of state tensors
                - actions: List of (action_type, action_tensor) tuples
                - rewards: List of reward tensors
        """
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards
        }