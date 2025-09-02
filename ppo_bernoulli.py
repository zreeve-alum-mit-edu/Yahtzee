import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
from torch.optim.lr_scheduler import LambdaLR
from device_config import device


class PolicyNetwork(nn.Module):
    """Policy network with Bernoulli hold decisions and categorical category selection."""
    
    def __init__(self, state_dim=39, hidden_dim=512, activation='relu', 
                 num_shared_layers=3, num_branch_layers=2):
        super(PolicyNetwork, self).__init__()
        
        # Shared backbone layers
        self.shared_layers = nn.ModuleList()
        for i in range(num_shared_layers):
            if i == 0:
                self.shared_layers.append(nn.Linear(state_dim, hidden_dim))
            else:
                self.shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Hold branch - separate hidden layers leading to hold head
        self.hold_branch = nn.ModuleList()
        for i in range(num_branch_layers):
            self.hold_branch.append(nn.Linear(hidden_dim, hidden_dim))
        # Hold action head - 5 independent Bernoulli decisions
        self.hold_head = nn.Linear(hidden_dim, 5)
        
        # Category branch - separate hidden layers leading to category head
        self.category_branch = nn.ModuleList()
        for i in range(num_branch_layers):
            self.category_branch.append(nn.Linear(hidden_dim, hidden_dim))
        # Category action head - 6 categories
        self.category_head = nn.Linear(hidden_dim, 6)
        
        # Store activation function and architecture params
        self.activation = activation
        self.num_shared_layers = num_shared_layers
        self.num_branch_layers = num_branch_layers
    
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
            return F.relu(x)  # Default to ReLU
    
    def forward(self, state, action_type):
        # Shared backbone layers
        x = state
        for layer in self.shared_layers:
            x = self.get_activation(layer(x))
        
        if action_type == 'hold':
            # Hold-specific branch
            h = x
            for layer in self.hold_branch:
                h = self.get_activation(layer(h))
            # Return 5 logits for independent Bernoulli decisions
            logits = self.hold_head(h)
        else:  # category
            # Category-specific branch
            c = x
            for layer in self.category_branch:
                c = self.get_activation(layer(c))
            logits = self.category_head(c)
        
        return logits


class ValueNetwork(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(self, state_dim=39, hidden_dim=512, activation='relu'):
        super(ValueNetwork, self).__init__()
        
        # Match the depth of PolicyNetwork
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
        # Store activation function
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
            return F.relu(x)  # Default to ReLU
    
    def forward(self, state):
        x = self.get_activation(self.fc1(state))
        x = self.get_activation(self.fc2(x))
        x = self.get_activation(self.fc3(x))
        x = self.get_activation(self.fc4(x))
        x = self.get_activation(self.fc5(x))
        value = self.fc6(x)
        return value


class PPOBernoulliPlayer:
    """PPO agent using Bernoulli distributions for hold decisions."""
    
    def __init__(self, lr=3e-4, hold_lr_mult=1.0, category_lr_mult=1.0, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, entropy_coef=0.01, batch_size=32, 
                 hidden_dim=256, num_hidden_layers=2, value_loss_coef=0.5,
                 max_grad_norm=None, activation='relu', gae_lambda=0.95,
                 num_shared_layers=3, num_branch_layers=2):
        self.device = device
        
        # Create networks with configurable architecture
        self.policy_net = PolicyNetwork(
            hidden_dim=hidden_dim, 
            activation=activation,
            num_shared_layers=num_shared_layers,
            num_branch_layers=num_branch_layers
        ).to(device)
        self.value_net = ValueNetwork(hidden_dim=hidden_dim, activation=activation).to(device)
        
        # Separate optimizers for different parts of the network
        # Shared backbone parameters
        shared_params = []
        for layer in self.policy_net.shared_layers:
            shared_params.extend(list(layer.parameters()))
        
        # Hold branch parameters (including branch layers and final head)
        hold_params = []
        for layer in self.policy_net.hold_branch:
            hold_params.extend(list(layer.parameters()))
        hold_params.extend(list(self.policy_net.hold_head.parameters()))
        
        # Category branch parameters (including branch layers and final head)
        category_params = []
        for layer in self.policy_net.category_branch:
            category_params.extend(list(layer.parameters()))
        category_params.extend(list(self.policy_net.category_head.parameters()))
        
        # Create parameter groups with different learning rates
        self.policy_optimizer = optim.Adam([
            {'params': shared_params, 'lr': lr},
            {'params': hold_params, 'lr': lr * hold_lr_mult},
            {'params': category_params, 'lr': lr * category_lr_mult}
        ])
        
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_shared_layers = num_shared_layers
        self.num_branch_layers = num_branch_layers
        self.hold_lr_mult = hold_lr_mult
        self.category_lr_mult = category_lr_mult
        
        # For storing old policy
        self.old_policy_net = PolicyNetwork(
            hidden_dim=hidden_dim, 
            activation=activation,
            num_shared_layers=num_shared_layers,
            num_branch_layers=num_branch_layers
        ).to(device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        # Learning rate schedulers (optional - can be activated by calling step())
        self.policy_scheduler = None
        self.value_scheduler = None
        self.total_updates = 0
        
        # Training mode flag
        self.training = True
        
    def decide_hold(self, game):
        """
        Decide which dice to hold using independent Bernoulli decisions.
        
        Args:
            game: SimpleYahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 5) with 0/1 values for hold decisions
        """
        # Get state
        state = self.get_state_from_game(game)
        
        # Get hold logits (5 independent decisions)
        with torch.no_grad():
            logits = self.policy_net(state, 'hold')  # (batch, 5)
            probs = torch.sigmoid(logits)  # Convert to probabilities
            
            if self.training:
                # Sample from Bernoulli distributions during training
                dist = Bernoulli(probs)
                hold_mask = dist.sample()  # (batch, 5) with 0/1 values
            else:
                # Use greedy selection during evaluation
                hold_mask = (probs > 0.5).float()  # Hold if prob > 0.5
        
        return hold_mask
    
    def decide_category(self, game):
        """
        Decide which category to score based on current game state.
        
        Args:
            game: SimpleYahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 1) with category index (0-5)
        """
        # Get state
        state = self.get_state_from_game(game)
        
        # Get category probabilities
        with torch.no_grad():
            logits = self.policy_net(state, 'category')
            
            if self.training:
                # Sample from categorical distribution during training
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                category = dist.sample().unsqueeze(-1)
            else:
                # Use greedy selection during evaluation
                category = torch.argmax(logits, dim=-1).unsqueeze(-1)
        
        return category
    
    def get_state_from_game(self, game):
        """Extract minimal state tensor from game instance."""
        # Flatten dice from (x, 5, 6) to (x, 30)
        dice_flat = game.dice.view(game.num_games, -1)
        
        # Extract only whether categories are open (x, 6) - using the 7th column which is 1 if unscored
        categories_open = game.scores[:, :, 6]  # Shape: (num_games, 6)
        
        # Turn is already (x, 3)
        
        # Concatenate all state components: dice (30) + categories_open (6) + turn (3) = 39
        state = torch.cat([dice_flat, categories_open, game.turn], dim=1)
        return state
    
    def compute_returns(self, rewards, dones, gamma=0.99):
        """Compute discounted returns."""
        returns = []
        discounted_reward = torch.zeros(rewards[0].shape[0], device=self.device)
        
        for t in reversed(range(len(rewards))):
            r_t = rewards[t].squeeze(-1)  # (B,) instead of (B,1)
            nonterminal = 0.0 if dones[t] else 1.0
            discounted_reward = r_t + gamma * nonterminal * discounted_reward
            returns.insert(0, discounted_reward)
        
        return torch.stack(returns)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE."""
        advantages = []
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            # Squeeze rewards to match values shape
            r_t = rewards[t].squeeze(-1)  # (B,) instead of (B,1)
            v_t = values[t]  # (B,)
            
            if t == len(rewards) - 1:
                next_value = torch.zeros_like(v_t)
            else:
                next_value = values[t + 1]
            
            # Zero out bootstrap at terminal
            nonterminal = 0.0 if dones[t] else 1.0
            
            delta = r_t + self.gamma * nonterminal * next_value - v_t
            last_advantage = delta + self.gamma * self.gae_lambda * nonterminal * last_advantage
            advantages.insert(0, last_advantage)
        
        return torch.stack(advantages)
    
    def train(self, trajectory):
        """
        Train the PPO agent using collected trajectory with mini-batch updates.
        
        Args:
            trajectory: Dictionary with 'states', 'actions', 'rewards' lists
        """
        # Update old policy BEFORE training (critical for PPO!)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        # Mark episode boundaries (every 18 steps = 1 episode)
        steps_per_episode = 18
        num_steps = len(states)  # This is 18 (number of timesteps)
        # Get the actual number of parallel games from the batch dimension
        num_games = states[0].shape[0]  # This is the batch size (e.g., 10000)
        dones = [(i + 1) % steps_per_episode == 0 for i in range(num_steps)]
        
        # Compute values for all states
        values = []
        for state in states:
            value = self.value_net(state)
            values.append(value)
        values = torch.stack(values).squeeze(-1)
        
        # Compute returns and advantages using GAE
        returns = self.compute_returns(rewards, dones)
        advantages = self.compute_advantages(rewards, values.detach(), dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update with mini-batches
        for _ in range(self.k_epochs):
            # Shuffle game order for this epoch (keep on GPU)
            torch.manual_seed(torch.randint(0, 10000, (1,)).item())  # Random seed for shuffling
            shuffled_games = torch.randperm(num_games, device=self.device)
            
            # Process in mini-batches of complete games
            for batch_start in range(0, num_games, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_games)
                
                # Slice the shuffled indices directly on GPU
                mini_batch_indices = shuffled_games[batch_start:batch_end]
                
                # Compute losses for this mini-batch
                policy_loss = 0
                value_loss = 0
                
                # Process each timestep
                for step_idx in range(num_steps):
                    # Get the full state/action for this timestep
                    state_full = states[step_idx]
                    action_type, action_full = actions[step_idx]
                    
                    # Extract only the games in this mini-batch
                    state = state_full[mini_batch_indices]
                    action = action_full[mini_batch_indices]
                    
                    # Policy loss
                    logits = self.policy_net(state, action_type)
                    
                    if action_type == 'hold':
                        # Bernoulli distribution for holds
                        probs = torch.sigmoid(logits)
                        dist = Bernoulli(probs)
                        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum over 5 dice
                        entropy = dist.entropy().sum(dim=-1)
                        
                        # Get old policy probability
                        with torch.no_grad():
                            old_logits = self.old_policy_net(state, action_type)
                            old_probs = torch.sigmoid(old_logits)
                            old_dist = Bernoulli(old_probs)
                            old_log_prob = old_dist.log_prob(action).sum(dim=-1)
                    else:  # category
                        # Categorical distribution for categories
                        probs = F.softmax(logits, dim=-1)
                        action_idx = action.squeeze(-1)
                        dist = Categorical(probs)
                        log_prob = dist.log_prob(action_idx)
                        entropy = dist.entropy()
                        
                        # Get old policy probability
                        with torch.no_grad():
                            old_logits = self.old_policy_net(state, action_type)
                            old_probs = F.softmax(old_logits, dim=-1)
                            old_dist = Categorical(old_probs)
                            old_log_prob = old_dist.log_prob(action_idx)
                    
                    # Compute ratio and clipped objective
                    ratio = torch.exp(log_prob - old_log_prob)
                    # Select advantages and returns for mini-batch
                    adv_batch = advantages[step_idx][mini_batch_indices]
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_batch
                    
                    # Add to policy loss
                    policy_loss -= torch.min(surr1, surr2).mean() + self.entropy_coef * entropy.mean()
                    
                    # Value loss
                    value_pred = self.value_net(state)
                    returns_batch = returns[step_idx][mini_batch_indices]
                    target = returns_batch.unsqueeze(-1) if returns_batch.dim() == 1 else returns_batch
                    value_loss += F.mse_loss(value_pred, target)
                
                # Average losses over timesteps only (already averaged over mini-batch via .mean())
                policy_loss = policy_loss / num_steps
                value_loss = value_loss / num_steps
                
                # Update networks for this mini-batch
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
        
        # Return losses from last mini-batch for logging
        return policy_loss.item(), value_loss.item()