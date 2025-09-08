from multi_yahtzee import MultiYahtzee
import torch


class GameRunner:
    def __init__(self, player, Z=1, device='cuda'):
        """
        Initialize GameRunner with a player object.
        
        Args:
            player: Player object that will make decisions during the game
            Z: Number of parallel Yahtzee variants to play
            device: Device to run on ('cuda' or 'cpu')
        """
        self.player = player
        self.Z = Z
        self.device = device
        # SAR tracking for PPO training
        self.states = []
        self.actions = []
        self.rewards = []
        self.available_masks = []
    
    def create_game(self, num_games):
        """
        Create a new Multi-Yahtzee game with specified number of parallel games.
        
        Args:
            num_games: Number of games to play in parallel
        """
        self.game = MultiYahtzee(num_games, Z=self.Z, device=self.device)
        # Clear SAR history when creating new game
        self.states = []
        self.actions = []
        self.rewards = []
        self.available_masks = []
    
    def get_state(self):
        """
        Get current game state from the player.
        
        Returns:
            Tensor containing game state
        """
        return self.player.get_state_from_game(self.game)
    
    def play_game(self):
        """
        Play a complete game of Multi-Yahtzee (13 rounds per Z game).
        Tracks states, actions, and rewards for PPO training.
        """
        # Clear the game state and SAR history
        self.game.clear()
        self.states = []
        self.actions = []
        self.rewards = []
        self.available_masks = []
        
        # Play 13 rounds per Z game = 13*Z total scoring decisions
        for round_num in range(13 * self.Z):
            # Update the round tracker for the current Z game
            z_idx = round_num // 13
            round_in_z = round_num % 13
            
            # Update round (shared across all Z - we track overall progress)
            self.game.round[:] = round_num
            
            # First roll - roll all dice
            self.game.roll_dice()
            
            # Set turn to first roll (all Z games share the same turn)
            self.game.turn.zero_()
            self.game.turn[:, 0, 0] = 1
            
            # First hold decision
            state = self.get_state()
            hold_mask = self.player.decide_hold(self.game)
            self.states.append(state)
            self.actions.append(('hold', hold_mask))
            # No immediate reward for hold decisions
            self.rewards.append(torch.zeros(self.game.num_games, 1, device=self.device))
            # No available mask for hold decisions
            self.available_masks.append(None)
            
            # Second roll - roll unheld dice
            self.game.roll_dice(hold_mask)
            
            # Set turn to second roll (all Z games share the same turn)
            self.game.turn.zero_()
            self.game.turn[:, 0, 1] = 1
            
            # Second hold decision
            state = self.get_state()
            hold_mask = self.player.decide_hold(self.game)
            self.states.append(state)
            self.actions.append(('hold', hold_mask))
            # No immediate reward for hold decisions
            self.rewards.append(torch.zeros(self.game.num_games, 1, device=self.device))
            # No available mask for hold decisions
            self.available_masks.append(None)
            
            # Third roll - roll unheld dice
            self.game.roll_dice(hold_mask)
            
            # Set turn to third roll (final) (all Z games share the same turn)
            self.game.turn.zero_()
            self.game.turn[:, 0, 2] = 1
            
            # Scoring decision
            state = self.get_state()
            # Get available categories mask for training
            available_mask = self.game.get_available_categories()
            category = self.player.decide_category(self.game)
            self.states.append(state)
            self.actions.append(('category', category))
            # Store available mask with category action for training
            self.available_masks.append(available_mask)
            
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
                - available_masks: List of available category masks (None for hold decisions)
        """
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'available_masks': self.available_masks
        }