import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
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


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (torch.stack(state),
                torch.tensor(action),
                torch.tensor(reward, dtype=torch.float32),
                torch.stack(next_state),
                torch.tensor(done, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


class DQNCategoryPlayer:
    """DQN agent for category selection with random hold decisions."""

    def __init__(self, Z=3, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, target_update_freq=100, batch_size=64,
                 hidden_dim=512, num_layers=4, activation='relu', buffer_size=100000,
                 use_double_dqn=True, device='cuda'):

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

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn

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

        if random.random() > epsilon:
            # Exploitation: choose best valid action based on Q-values
            with torch.no_grad():
                q_values = self.q_network(state)

                # Mask invalid actions with -inf
                q_values[~valid_mask] = float('-inf')

                actions = q_values.argmax(dim=1)
        else:
            # Exploration: choose random valid action
            actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            for i in range(batch_size):
                valid_indices = torch.where(valid_mask[i])[0]
                if len(valid_indices) > 0:
                    actions[i] = valid_indices[torch.randint(len(valid_indices), (1,))]

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

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        """Perform one step of DQN training."""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

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

            # Compute target values
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1

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