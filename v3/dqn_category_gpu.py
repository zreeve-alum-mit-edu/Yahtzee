import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from multi_yahtzee import MultiYahtzee
from precomputes import (
    load_w_full_torch,
    dice_onehot_to_state_id_vectorized,
    get_keep_masks_tensor,
    create_all_sorted_dice_onehot
)

# Constants for dice holding patterns (32 possible combinations of 5 dice)
DICE_HOLD_MASKS = torch.tensor([
    [int(x) for x in format(i, '05b')] for i in range(32)
], dtype=torch.float32)


class QNetwork(nn.Module):
    """Q-Network for estimating Q(s,a) values for category selection."""

    def __init__(self, state_dim, num_actions, hidden_dim=512, num_layers=4, activation='relu'):
        super(QNetwork, self).__init__()

        self.layers = nn.ModuleList()

        # Build hidden layers
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(state_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_actions)

        self.activation = activation

    def get_activation(self, x):
        """Apply the configured activation function."""
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'leaky_relu':
            return F.leaky_relu(x, 0.01)
        elif self.activation == 'elu':
            return F.elu(x)
        elif self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'gelu':
            return F.gelu(x)
        else:
            return F.relu(x)

    def forward(self, state):
        """
        Forward pass through Q-network.

        Args:
            state: Tensor of shape [batch_size, state_dim]

        Returns:
            Q-values for all actions [batch_size, num_actions]
        """
        x = state
        for layer in self.layers:
            x = self.get_activation(layer(x))
        q_values = self.output_layer(x)
        return q_values


class GPUReplayBuffer:
    """Fully GPU-based ring buffer for experience replay."""

    def __init__(self, capacity, state_dim, device='cuda', use_fp16=True):
        """
        Initialize GPU replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state vectors
            device: Device to store buffer on
            use_fp16: Whether to use float16 for states (saves memory)
        """
        self.capacity = capacity
        self.device = device
        self.state_dtype = torch.float16 if use_fp16 else torch.float32

        # Preallocate tensors on GPU
        self.states = torch.empty((capacity, state_dim), device=device, dtype=self.state_dtype)
        self.actions = torch.empty((capacity,), device=device, dtype=torch.long)
        self.rewards = torch.empty((capacity,), device=device, dtype=torch.float32)
        self.next_states = torch.empty((capacity, state_dim), device=device, dtype=self.state_dtype)
        self.dones = torch.empty((capacity,), device=device, dtype=torch.bool)

        # Ring buffer pointers
        self.write_ptr = 0
        self.size = 0

    def push(self, states, actions, rewards, next_states, dones):
        """
        Add batch of transitions to the buffer.

        Args:
            All arguments should be tensors on the same device as buffer
        """
        batch_size = states.shape[0]

        # Calculate indices for writing
        if self.write_ptr + batch_size <= self.capacity:
            # Simple case: no wrap-around
            indices = torch.arange(self.write_ptr, self.write_ptr + batch_size, device=self.device)
        else:
            # Handle wrap-around
            indices1 = torch.arange(self.write_ptr, self.capacity, device=self.device)
            indices2 = torch.arange(0, (self.write_ptr + batch_size) % self.capacity, device=self.device)
            indices = torch.cat([indices1, indices2])

        # Write to buffer (convert states to fp16 if needed)
        if self.state_dtype == torch.float16:
            self.states[indices] = states.to(torch.float16)
            self.next_states[indices] = next_states.to(torch.float16)
        else:
            self.states[indices] = states
            self.next_states[indices] = next_states

        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.dones[indices] = dones

        # Update pointers
        self.write_ptr = (self.write_ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of transitions.

        Returns:
            Tuple of tensors, all on GPU
        """
        # Sample random indices
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Gather samples (convert states back to fp32 for network)
        states = self.states[indices].to(torch.float32)
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices].to(torch.float32)
        dones = self.dones[indices].to(torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


class DQNCategoryPlayerGPU:
    """DQN agent with fully GPU-based replay buffer."""

    def __init__(self, Z=3, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay_steps=10000, target_update_freq=1, batch_size=64,
                 hidden_dim=512, num_layers=4, activation='relu', buffer_size=100000,
                 use_double_dqn=True, use_huber_loss=True, use_fp16_buffer=True,
                 device='cuda'):

        self.device = device
        self.Z = Z
        self.num_categories = Z * 13  # Total categories across all Z games

        # Calculate state dimension for Multi-Yahtzee
        # state_dim = 30 (dice) + Z*42 (upper counts) + Z*7 (lower status only) + 3 (turn) = 33 + Z*49
        self.state_dim = 33 + Z * 49

        # Q-networks
        self.q_network = QNetwork(self.state_dim, self.num_categories, hidden_dim,
                                 num_layers, activation).to(device)
        self.target_network = QNetwork(self.state_dim, self.num_categories, hidden_dim,
                                      num_layers, activation).to(device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # GPU Replay buffer
        self.memory = GPUReplayBuffer(buffer_size, self.state_dim, device, use_fp16_buffer)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.use_huber_loss = use_huber_loss

        # Training counters
        self.steps_done = 0
        self.episodes_done = 0

        # Move hold masks to device
        self.dice_hold_masks = DICE_HOLD_MASKS.to(device)

        # Cache for round computations (tuple of tensors, all on GPU)
        self.round_cache = None

    def get_state(self, game):
        """Extract state representation from Multi-Yahtzee game."""
        batch_size = game.num_games

        # Flatten dice (5 dice x 6 faces = 30)
        dice_flat = game.dice.view(batch_size, -1)

        # Upper section: use counts directly (Z games x 6 categories x 7 scores = Z*42)
        upper_flat = game.upper.view(batch_size, -1)

        # Lower section: use only status (scored/unscored) (Z games x 7 categories x 1 = Z*7)
        lower_status = game.lower[:, :, :, 1].view(batch_size, -1)

        # Turn indicator (3 values, one-hot encoded)
        # Use first Z dimension since all share same dice
        turn_flat = game.turn[:, 0, :].view(batch_size, -1)

        # Concatenate all features
        state = torch.cat([dice_flat, upper_flat, lower_status, turn_flat], dim=1)

        return state

    def select_hold_action(self, state, game):
        """
        Select hold action randomly.

        Args:
            state: Current state tensor
            game: MultiYahtzee game instance

        Returns:
            Hold mask tensor of shape [batch_size, 5]
        """
        batch_size = state.shape[0]

        # Random hold patterns (0-31)
        random_actions = torch.randint(0, 32, (batch_size,), device=self.device)

        # Convert to hold masks
        hold_masks = self.dice_hold_masks[random_actions]

        return hold_masks

    def select_category_action(self, state, game, epsilon=None):
        """
        Select category action using epsilon-greedy policy.

        Args:
            state: Current state tensor
            game: MultiYahtzee game instance
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected category indices
        """
        batch_size = state.shape[0]

        if epsilon is None:
            epsilon = self.epsilon

        # Get valid category mask
        valid_mask = self.get_valid_category_mask(game)

        # Create random mask for epsilon-greedy
        random_mask = torch.rand(batch_size, device=self.device) < epsilon

        # Get Q-values for all games
        with torch.no_grad():
            q_values = self.q_network(state)
            # Mask invalid actions with -inf
            q_values[~valid_mask] = float('-inf')
            greedy_actions = q_values.argmax(dim=1)

        # Get random valid actions
        random_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            valid_indices = torch.where(valid_mask[i])[0]
            if len(valid_indices) > 0:
                random_actions[i] = valid_indices[torch.randint(len(valid_indices), (1,), device=self.device)]

        # Combine greedy and random actions based on epsilon
        actions = torch.where(random_mask, random_actions, greedy_actions)

        return actions

    def get_valid_category_mask(self, game):
        """
        Get mask of valid categories (unscored categories).

        Returns:
            Boolean tensor of shape [batch_size, Z*13]
        """
        batch_size = game.num_games

        # Upper section: check if unscored (last element = 1)
        upper_unscored = game.upper[:, :, :, 6] == 1  # [batch_size, Z, 6]

        # Lower section: check if unscored
        lower_unscored = game.lower[:, :, :, 1] == 1  # [batch_size, Z, 7]

        # Combine upper and lower for each scorecard
        valid_categories = []
        for z in range(self.Z):
            # Combine upper[z] and lower[z]
            valid_z = torch.cat([upper_unscored[:, z], lower_unscored[:, z]], dim=1)
            valid_categories.append(valid_z)

        # Stack all Z scorecards
        valid_mask = torch.cat(valid_categories, dim=1)  # [batch_size, Z*13]

        return valid_mask

    def store_transitions(self, states, actions, rewards, next_states, dones):
        """Store batch of transitions in the GPU replay buffer."""
        # Ensure all tensors are on the correct device
        if states.device != self.device:
            states = states.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        if rewards.device != self.device:
            rewards = rewards.to(self.device)
        if next_states.device != self.device:
            next_states = next_states.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)

        self.memory.push(states, actions, rewards, next_states, dones)

    def update(self):
        """Perform one step of DQN training."""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from GPU replay buffer (already on device)
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_network(next_states)
                next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(dim=1)[0]

            # Compute target values (don't bootstrap from terminal states)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        if self.use_huber_loss:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        else:
            loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update epsilon (decay per step, not per episode)
        self.update_epsilon()

        # Increment steps (for epsilon decay)
        self.steps_done += 1

        return loss.item()

    def update_epsilon(self):
        """Update epsilon using linear decay based on steps."""
        if self.steps_done < self.epsilon_decay_steps:
            # Linear decay
            decay_progress = self.steps_done / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        else:
            self.epsilon = self.epsilon_end

    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath):
        """Save the model and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'Z': self.Z,
            'state_dim': self.state_dim,
            'buffer_size': len(self.memory),
            'buffer_capacity': self.memory.capacity,
        }, filepath)

    def load(self, filepath):
        """Load the model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)

    def compute_V3_from_Q(self, games, W_full=None):
        """
        Compute V3 (value at S3) for all 252 dice states under current meta-state.
        Fully vectorized version - processes all games and dice states in one forward pass.

        Args:
            games: MultiYahtzee instance with current board state
            W_full: Optional preloaded transition tensor (will load if None)

        Returns:
            V3_all: Tensor [B, 252] of values for each dice state per game
        """
        batch_size = games.num_games
        device = self.device

        # Create all 252 sorted dice states once
        all_dice_onehot = create_all_sorted_dice_onehot(device)  # [252, 5, 6]
        dice_flat = all_dice_onehot.view(252, -1)  # [252, 30]

        # Repeat dice for all games: [B*252, 30]
        dice_repeated = dice_flat.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 30)

        # Expand board features for all 252 dice states per game
        # Upper: [B, Z, 6, 7] -> [B, 1, Z*42] -> [B, 252, Z*42] -> [B*252, Z*42]
        upper_flat = games.upper.view(batch_size, -1).unsqueeze(1).expand(-1, 252, -1).reshape(-1, self.Z * 42)

        # Lower status: [B, Z, 7, 2] -> [B, Z*7] -> [B, 1, Z*7] -> [B, 252, Z*7] -> [B*252, Z*7]
        lower_status = games.lower[:, :, :, 1].view(batch_size, -1).unsqueeze(1).expand(-1, 252, -1).reshape(-1, self.Z * 7)

        # Turn indicator for S3 (third roll): [B*252, 3]
        turn_s3 = torch.zeros(batch_size * 252, 3, device=device)
        turn_s3[:, 2] = 1  # Set all to third roll

        # Concatenate all features into one big state tensor: [B*252, state_dim]
        states_all = torch.cat([dice_repeated, upper_flat, lower_status, turn_s3], dim=1)

        # Single forward pass through Q-network
        with torch.no_grad():
            q_values_all = self.q_network(states_all)  # [B*252, Z*13]

            # Build valid category mask for all games
            # Upper unscored: [B, Z, 6] where last dim indicates if unscored
            upper_unscored = games.upper[:, :, :, 6] == 1  # [B, Z, 6]
            # Lower unscored: [B, Z, 7]
            lower_unscored = games.lower[:, :, :, 1] == 1  # [B, Z, 7]

            # Combine into full mask per game: [B, Z*13]
            valid_masks = []
            for z in range(self.Z):
                valid_z = torch.cat([upper_unscored[:, z], lower_unscored[:, z]], dim=1)  # [B, 13]
                valid_masks.append(valid_z)
            valid_mask = torch.cat(valid_masks, dim=1)  # [B, Z*13]

            # Repeat each game's mask 252 times: [B*252, Z*13]
            valid_mask_expanded = valid_mask.repeat_interleave(252, dim=0)

            # Mask invalid categories with -inf
            q_values_all[~valid_mask_expanded] = float('-inf')

            # Take max over categories and reshape back to [B, 252]
            V3_all = q_values_all.max(dim=1)[0].view(batch_size, 252)

        return V3_all

    def compute_round_cache(self, games, W_full=None):
        """
        Precompute and cache all values needed for a round.
        This includes V3, V2, V1, best hold masks, and best categories.

        Args:
            games: MultiYahtzee instance with current board state
            W_full: Preloaded transition tensor (will load if None)

        Returns:
            Dictionary with cached values for the round
        """
        device = self.device
        batch_size = games.num_games

        # Load W_full if not provided
        if W_full is None:
            W_full = load_w_full_torch(device=device)

        # Step 1: Compute V3 and best categories for all 252 dice states
        # Create all 252 sorted dice states once
        all_dice_onehot = create_all_sorted_dice_onehot(device)  # [252, 5, 6]
        dice_flat = all_dice_onehot.view(252, -1)  # [252, 30]

        # Repeat dice for all games: [B*252, 30]
        dice_repeated = dice_flat.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 30)

        # Expand board features for all 252 dice states per game
        upper_flat = games.upper.view(batch_size, -1).unsqueeze(1).expand(-1, 252, -1).reshape(-1, self.Z * 42)
        lower_status = games.lower[:, :, :, 1].view(batch_size, -1).unsqueeze(1).expand(-1, 252, -1).reshape(-1, self.Z * 7)

        # Turn indicator for S3 (third roll)
        turn_s3 = torch.zeros(batch_size * 252, 3, device=device)
        turn_s3[:, 2] = 1

        # Build state tensor: [B*252, state_dim]
        states_all = torch.cat([dice_repeated, upper_flat, lower_status, turn_s3], dim=1)

        # Single forward pass through Q-network
        with torch.no_grad():
            q_values_all = self.q_network(states_all)  # [B*252, Z*13]

            # Build valid category mask for all games
            upper_unscored = games.upper[:, :, :, 6] == 1  # [B, Z, 6]
            lower_unscored = games.lower[:, :, :, 1] == 1  # [B, Z, 7]

            # Combine into full mask per game: [B, Z*13]
            valid_masks = []
            for z in range(self.Z):
                valid_z = torch.cat([upper_unscored[:, z], lower_unscored[:, z]], dim=1)
                valid_masks.append(valid_z)
            valid_mask = torch.cat(valid_masks, dim=1)  # [B, Z*13]

            # Repeat each game's mask 252 times: [B*252, Z*13]
            valid_mask_expanded = valid_mask.repeat_interleave(252, dim=0)

            # Mask invalid categories with -inf
            q_values_masked = q_values_all.clone()
            q_values_masked[~valid_mask_expanded] = float('-inf')

            # Get best categories and values for each dice state
            V3_all, best_categories = q_values_masked.max(dim=1)  # [B*252], [B*252]
            V3_all = V3_all.view(batch_size, 252)  # [B, 252]
            best_categories = best_categories.view(batch_size, 252)  # [B, 252]

        # Step 2: Backward induction V3 → V2 → V1
        # S2 from S3
        q2_all = torch.einsum("hkp,bp->bhk", W_full, V3_all)  # [B, 252, 32]
        V2_all, best_k_s2 = q2_all.max(dim=2)  # [B, 252], [B, 252]

        # S1 from S2
        q1_all = torch.einsum("hkp,bp->bhk", W_full, V2_all)  # [B, 252, 32]
        V1_all, best_k_s1 = q1_all.max(dim=2)  # [B, 252], [B, 252]

        # Cache all computed values as a tuple (all tensors stay on GPU)
        # Order: (V3_all, V2_all, V1_all, q2_all, q1_all, best_k_s2, best_k_s1, best_categories)
        cache = (
            V3_all,           # 0: [B, 252]
            V2_all,           # 1: [B, 252]
            V1_all,           # 2: [B, 252]
            q2_all,           # 3: [B, 252, 32]
            q1_all,           # 4: [B, 252, 32]
            best_k_s2,        # 5: [B, 252]
            best_k_s1,        # 6: [B, 252]
            best_categories,  # 7: [B, 252]
        )

        # Store in instance cache
        self.round_cache = cache
        return cache

    def get_cached_hold_action(self, games, roll_num, cache=None):
        """
        Get optimal hold action using cached values.

        Args:
            games: MultiYahtzee instance
            roll_num: Which roll (1 or 2)
            cache: Precomputed cache tuple (uses self.round_cache if None)

        Returns:
            Hold mask tensor of shape [B, 5]
        """
        if cache is None:
            cache = self.round_cache

        # Get current dice state indices
        current_dice = games.dice  # [B, 5, 6]
        h_current = dice_onehot_to_state_id_vectorized(current_dice)  # [B]

        # Get best hold mask based on roll number
        # Cache indices: 6=best_k_s1, 5=best_k_s2
        if roll_num == 1:
            # First roll: use S1 holds (index 6)
            best_k = torch.gather(cache[6], 1, h_current.unsqueeze(1)).squeeze(1)  # [B]
        elif roll_num == 2:
            # Second roll: use S2 holds (index 5)
            best_k = torch.gather(cache[5], 1, h_current.unsqueeze(1)).squeeze(1)  # [B]
        else:
            raise ValueError(f"Invalid roll_num: {roll_num}, must be 1 or 2")

        # Convert k indices to hold masks
        keep_masks = get_keep_masks_tensor(games.device)  # [32, 5]
        hold_masks = keep_masks[best_k]  # [B, 5]

        return hold_masks

    def get_cached_category_action(self, games, cache=None):
        """
        Get optimal category action using cached values.

        Args:
            games: MultiYahtzee instance
            cache: Precomputed cache tuple (uses self.round_cache if None)

        Returns:
            Category indices tensor of shape [B]
        """
        if cache is None:
            cache = self.round_cache

        # Get current dice state indices (after final roll)
        current_dice = games.dice  # [B, 5, 6]
        h_current = dice_onehot_to_state_id_vectorized(current_dice)  # [B]

        # Get best category for current dice (index 7)
        best_categories = torch.gather(cache[7], 1, h_current.unsqueeze(1)).squeeze(1)  # [B]

        return best_categories

    def select_optimal_hold_action(self, state, games, W_full=None):
        """
        Select optimal hold action using backward induction with Q-values.

        Args:
            state: Current state tensor [B, state_dim]
            games: MultiYahtzee game instance
            W_full: Optional preloaded transition tensor (will load if None)

        Returns:
            Hold mask tensor of shape [B, 5] (1=keep, 0=reroll)
        """
        batch_size = state.shape[0]
        device = self.device

        # Load W_full if not provided
        if W_full is None:
            W_full = load_w_full_torch(device=device)  # [252, 32, 252]

        # Step 1: Compute V3 for all 252 dice states
        V3_all = self.compute_V3_from_Q(games, W_full)  # [B, 252]

        # Step 2: Backward induction V3 → V2 → V1
        # S2 from S3: for each game b
        q2_all = torch.einsum("hkp,bp->bhk", W_full, V3_all)  # [B, 252, 32]
        V2_all, best_k_s2 = q2_all.max(dim=2)  # [B, 252], [B, 252]

        # S1 from S2: for each game b
        q1_all = torch.einsum("hkp,bp->bhk", W_full, V2_all)  # [B, 252, 32]
        V1_all, best_k_s1 = q1_all.max(dim=2)  # [B, 252], [B, 252]

        # Step 3: Get current dice state indices
        current_dice = games.dice  # [B, 5, 6]
        h0_all = dice_onehot_to_state_id_vectorized(current_dice)  # [B]

        # Step 4: Get best hold mask for each game's current state
        best_k = torch.gather(best_k_s1, 1, h0_all.unsqueeze(1)).squeeze(1)  # [B]

        # Convert k indices to hold masks
        keep_masks = get_keep_masks_tensor(device)  # [32, 5]
        hold_masks = keep_masks[best_k]  # [B, 5]

        return hold_masks


def choose_best_S1_hold_from_Q(games, player, W_full=None, device="cuda"):
    """
    Standalone function to choose best S1 hold using Q-network values.

    Args:
        games: MultiYahtzee instance
        player: DQNCategoryPlayerGPU instance
        W_full: Optional preloaded transition tensor
        device: Device to use

    Returns:
        Tensor [B, 5] of keep masks (1=keep, 0=reroll)
    """
    # Get current state
    state = player.get_state(games)

    # Use player's method to get optimal holds
    return player.select_optimal_hold_action(state, games, W_full)