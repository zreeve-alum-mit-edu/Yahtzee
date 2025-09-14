import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from multi_yahtzee import MultiYahtzee

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
                 epsilon_decay_steps=10000, target_update_freq=100, batch_size=64,
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

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def update_epsilon(self):
        """Update epsilon using linear decay based on steps."""
        if self.steps_done < self.epsilon_decay_steps:
            # Linear decay
            decay_progress = self.steps_done / self.epsilon_decay_steps
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_progress
        else:
            self.epsilon = self.epsilon_end

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