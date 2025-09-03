import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
from torch.optim.lr_scheduler import LambdaLR
from device_config import device
from yahtzee import Yahtzee


class PolicyNetwork(nn.Module):
    """Policy network with Bernoulli hold decisions and categorical category selection."""
    
    def __init__(self, state_dim=82, hidden_dim=512, activation='relu', 
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
        # Category action head - 13 categories (6 upper + 7 lower)
        self.category_head = nn.Linear(hidden_dim, 13)
        
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
    
    def __init__(self, state_dim=82, hidden_dim=512, activation='relu'):
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
                 num_shared_layers=3, num_branch_layers=2, use_amp=True, use_compile=True):
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_compile = use_compile
        
        # Create networks with configurable architecture
        self.policy_net = PolicyNetwork(
            hidden_dim=hidden_dim, 
            activation=activation,
            num_shared_layers=num_shared_layers,
            num_branch_layers=num_branch_layers
        ).to(device)
        self.value_net = ValueNetwork(hidden_dim=hidden_dim, activation=activation).to(device)
        
        # Compile networks for optimization if requested and available
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
                self.policy_net = torch.compile(self.policy_net, mode='default')
                self.value_net = torch.compile(self.value_net, mode='default')
                print("Networks compiled with torch.compile")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
                print("Continuing without compilation")
        
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
        # Don't compile old_policy_net as it's only used for inference
        # Copy initial weights
        self._copy_to_old_policy()
        
        # AMP GradScaler for mixed precision training
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("AMP (Automatic Mixed Precision) enabled")
        else:
            self.scaler = None
        
        # Learning rate schedulers (optional - can be activated by calling step())
        self.policy_scheduler = None
        self.value_scheduler = None
        self.total_updates = 0
        
        # Training mode flag
        self.training = True
    
    def _copy_to_old_policy(self):
        """Helper to copy state dict from compiled model to old policy."""
        if self.use_compile and hasattr(torch, 'compile'):
            # Compiled models have _orig_mod prefix, need to strip it
            state_dict = self.policy_net.state_dict()
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    clean_state_dict[key[10:]] = value  # Remove '_orig_mod.' prefix
                else:
                    clean_state_dict[key] = value
            self.old_policy_net.load_state_dict(clean_state_dict)
        else:
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())
    
    def decide_hold(self, game):
        """
        Decide which dice to hold using independent Bernoulli decisions.
        
        Args:
            game: Yahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 5) with 0/1 values for hold decisions
        """
        # Get state
        state = self.get_state_from_game(game)
        
        # Get hold logits (5 independent decisions)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.policy_net(state, 'hold')  # (batch, 5)
                probs = torch.sigmoid(logits)  # Convert to probabilities
            
            if self.training:
                # Sample from Bernoulli distributions during training
                dist = Bernoulli(probs)
                hold_mask = dist.sample().bool().float()  # (batch, 5) with 0/1 values as float
            else:
                # Use greedy selection during evaluation
                hold_mask = (probs > 0.5).float()  # Hold if prob > 0.5
        
        return hold_mask
    
    def decide_category(self, game):
        """
        Decide which category to score based on current game state.
        
        Args:
            game: Yahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 1) with category index (0-12)
        """
        # Get state
        state = self.get_state_from_game(game)
        
        # Get category probabilities
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.use_amp):
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
        """Extract state tensor from v2 Yahtzee game instance."""
        # Flatten dice from (x, 5, 6) to (x, 30)
        dice_flat = game.dice.view(game.num_games, -1)
        
        # Upper scores: extract full information needed for bonus tracking
        # Shape: (num_games, 6, 7) -> flatten to (num_games, 42)
        # This includes counts and scored/unscored status for all upper categories
        upper_flat = game.upper_scores.view(game.num_games, -1)
        
        # Lower scores: extract whether categories are open
        # Shape: (num_games, 7) - 1 if unscored, 0 if scored
        lower_open = game.lower_scores[:, :, 1]
        
        # State only includes relevant decision-making information:
        # dice (30) + upper (42) + lower_open (7) + turn (3) = 82
        state = torch.cat([
            dice_flat,           # 30
            upper_flat,          # 42  
            lower_open,          # 7
            game.turn,           # 3
        ], dim=1)               # Total: 82
        
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
        self._copy_to_old_policy()
        
        states = trajectory['states']
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        
        # Mark episode boundaries (every 39 steps = 1 episode)
        steps_per_episode = 39  # 13 rounds × 3 decisions
        num_steps = len(states)  # This is 39 (number of timesteps)
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
    
    def train_flattened(self, trajectory):
        """
        Train PPO using flattened trajectory for maximum efficiency.
        Processes all hold decisions together and all category decisions together.
        
        Args:
            trajectory: Dictionary with 'states', 'actions', 'rewards' lists
        """
        # Update old policy BEFORE training (critical for PPO!)
        self._copy_to_old_policy()
        
        states = trajectory['states']
        actions = trajectory['actions'] 
        rewards = trajectory['rewards']
        
        # Get dimensions
        T = len(states)  # 39 timesteps (13 rounds × 3 decisions)
        B = states[0].shape[0]  # e.g., 10000 games
        steps_per_episode = 39
        
        # Stack for easier manipulation
        S = torch.stack(states)  # (T, B, 82)
        R = torch.stack(rewards).squeeze(-1)  # (T, B)
        
        # Compute values for all states efficiently
        with torch.no_grad():
            V = torch.stack([self.value_net(s).squeeze(-1) for s in states])  # (T, B)
        
        # Mark episode boundaries  
        dones = [(i + 1) % steps_per_episode == 0 for i in range(T)]
        
        # Compute returns and advantages using GAE
        returns = self.compute_returns(rewards, dones)  # (T, B)
        advantages = self.compute_advantages(rewards, V, dones)  # (T, B)
        
        # Identify hold vs category timesteps
        is_hold = torch.tensor([a[0] == 'hold' for a in actions], device=self.device)
        is_cat = ~is_hold
        
        # Build flattened datasets for holds
        hold_indices = is_hold.nonzero().squeeze(-1).tolist()
        if hold_indices:
            S_hold = torch.cat([states[i] for i in hold_indices], dim=0)  # (26*B, 82)
            A_hold = torch.cat([actions[i][1] for i in hold_indices], dim=0)  # (26*B, 5)
            R_hold = torch.cat([returns[i] for i in hold_indices], dim=0)  # (26*B,)
            Adv_hold = torch.cat([advantages[i] for i in hold_indices], dim=0)  # (26*B,)
            
            # Normalize advantages per type
            Adv_hold = (Adv_hold - Adv_hold.mean()) / (Adv_hold.std() + 1e-8)
            
            # Precompute old log probs for holds
            with torch.no_grad():
                old_logits_h = self.old_policy_net(S_hold, 'hold')
                old_probs_h = torch.sigmoid(old_logits_h)
                old_dist_h = Bernoulli(old_probs_h)
                old_logp_h = old_dist_h.log_prob(A_hold).sum(dim=-1)
        
        # Build flattened datasets for categories
        cat_indices = is_cat.nonzero().squeeze(-1).tolist()
        if cat_indices:
            S_cat = torch.cat([states[i] for i in cat_indices], dim=0)  # (13*B, 82)
            A_cat = torch.cat([actions[i][1] for i in cat_indices], dim=0)  # (13*B, 1)
            R_cat = torch.cat([returns[i] for i in cat_indices], dim=0)  # (13*B,)
            Adv_cat = torch.cat([advantages[i] for i in cat_indices], dim=0)  # (13*B,)
            
            # Normalize advantages per type
            Adv_cat = (Adv_cat - Adv_cat.mean()) / (Adv_cat.std() + 1e-8)
            
            # Precompute old log probs for categories
            with torch.no_grad():
                old_logits_c = self.old_policy_net(S_cat, 'category')
                old_probs_c = F.softmax(old_logits_c, dim=-1)
                old_dist_c = Categorical(old_probs_c)
                old_logp_c = old_dist_c.log_prob(A_cat.squeeze(-1))
        
        # Large batch size for better GPU utilization
        mb_size = min(2048, B * 6)  # At least full games worth of data
        
        # Track losses for logging
        total_policy_loss = 0
        total_value_loss = 0
        num_updates = 0
        
        # PPO epochs
        for _ in range(self.k_epochs):
            
            # Process hold decisions
            if hold_indices:
                Nh = S_hold.shape[0]
                perm_h = torch.randperm(Nh, device=self.device)
                
                for i in range(0, Nh, mb_size):
                    idx = perm_h[i:min(i + mb_size, Nh)]
                    
                    # Use AMP autocast for forward pass
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        # Policy forward pass
                        logits = self.policy_net(S_hold[idx], 'hold')
                        probs = torch.sigmoid(logits)
                        dist = Bernoulli(probs)
                        logp = dist.log_prob(A_hold[idx]).sum(dim=-1)
                        ent = dist.entropy().sum(dim=-1)
                        
                        # PPO loss
                        ratio = torch.exp(logp - old_logp_h[idx])
                        surr1 = ratio * Adv_hold[idx]
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * Adv_hold[idx]
                        policy_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * ent.mean())
                        
                        # Value loss
                        vpred = self.value_net(S_hold[idx]).squeeze(-1)
                        value_loss = F.mse_loss(vpred, R_hold[idx])
                        
                        # Combined loss
                        total_loss = policy_loss + self.value_loss_coef * value_loss
                    
                    # Backward pass with AMP scaling
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    
                    if self.use_amp:
                        self.scaler.scale(total_loss).backward()
                        
                        if self.max_grad_norm is not None:
                            self.scaler.unscale_(self.policy_optimizer)
                            self.scaler.unscale_(self.value_optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                        
                        self.scaler.step(self.policy_optimizer)
                        self.scaler.step(self.value_optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                        
                        self.policy_optimizer.step()
                        self.value_optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    num_updates += 1
            
            # Process category decisions
            if cat_indices:
                Nc = S_cat.shape[0]
                perm_c = torch.randperm(Nc, device=self.device)
                
                for i in range(0, Nc, mb_size):
                    idx = perm_c[i:min(i + mb_size, Nc)]
                    
                    # Use AMP autocast for forward pass
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        # Policy forward pass
                        logits = self.policy_net(S_cat[idx], 'category')
                        probs = F.softmax(logits, dim=-1)
                        dist = Categorical(probs)
                        logp = dist.log_prob(A_cat[idx].squeeze(-1))
                        ent = dist.entropy()
                        
                        # PPO loss
                        ratio = torch.exp(logp - old_logp_c[idx])
                        surr1 = ratio * Adv_cat[idx]
                        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * Adv_cat[idx]
                        policy_loss = -(torch.min(surr1, surr2).mean() + self.entropy_coef * ent.mean())
                        
                        # Value loss
                        vpred = self.value_net(S_cat[idx]).squeeze(-1)
                        value_loss = F.mse_loss(vpred, R_cat[idx])
                        
                        # Combined loss
                        total_loss = policy_loss + self.value_loss_coef * value_loss
                    
                    # Backward pass with AMP scaling
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    
                    if self.use_amp:
                        self.scaler.scale(total_loss).backward()
                        
                        if self.max_grad_norm is not None:
                            self.scaler.unscale_(self.policy_optimizer)
                            self.scaler.unscale_(self.value_optimizer)
                            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                        
                        self.scaler.step(self.policy_optimizer)
                        self.scaler.step(self.value_optimizer)
                        self.scaler.update()
                    else:
                        total_loss.backward()
                        
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                        
                        self.policy_optimizer.step()
                        self.value_optimizer.step()
                    
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    num_updates += 1
        
        # Return average losses
        avg_policy_loss = total_policy_loss / max(num_updates, 1)
        avg_value_loss = total_value_loss / max(num_updates, 1)
        
        return avg_policy_loss, avg_value_loss