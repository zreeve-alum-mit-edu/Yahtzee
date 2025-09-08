import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from multi_yahtzee import MultiYahtzee

# Constants for dice holding patterns (32 possible combinations of 5 dice)
DICE_HOLD_MASKS = torch.tensor([
    [int(x) for x in format(i, '05b')] for i in range(32)
], dtype=torch.float32)


class PolicyNetwork(nn.Module):
    """Policy network with categorical hold decisions (32 actions) and categorical category selection."""
    
    def __init__(self, state_dim, Z=3, hidden_dim=512, activation='relu', 
                 num_shared_layers=3, num_branch_layers=2):
        super(PolicyNetwork, self).__init__()
        
        self.Z = Z
        self.num_categories = Z * 13  # Total categories across all Z games
        
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
        # Hold action head - 32 discrete actions (2^5 hold patterns)
        self.hold_head = nn.Linear(hidden_dim, 32)
        
        # Category branch - separate hidden layers leading to category head
        self.category_branch = nn.ModuleList()
        for i in range(num_branch_layers):
            self.category_branch.append(nn.Linear(hidden_dim, hidden_dim))
        # Category action head - Z*13 categories
        self.category_head = nn.Linear(hidden_dim, self.num_categories)
        
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
            # Return 32 logits for categorical hold pattern selection
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
    
    def __init__(self, state_dim, hidden_dim=512, activation='relu'):
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


class PPOHoldPlayer:
    """PPO agent using categorical distributions for hold decisions (32 discrete actions)."""
    
    def __init__(self, Z=3, lr=3e-4, hold_lr_mult=1.0, category_lr_mult=1.0, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, entropy_coef=0.01, batch_size=32, 
                 hidden_dim=256, num_hidden_layers=2, value_loss_coef=0.5,
                 max_grad_norm=None, activation='relu', gae_lambda=0.95,
                 num_shared_layers=3, num_branch_layers=2, use_amp=True, use_compile=True,
                 device='cuda'):
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_compile = use_compile
        self.Z = Z
        
        # Calculate state dimension for Multi-Yahtzee
        # state_dim = 30 (dice) + Z*42 (upper counts) + Z*7 (lower status only) + 3 (turn) = 33 + Z*49
        self.state_dim = 33 + Z * 49
        
        # Load precalculated hold masks
        self.hold_masks = DICE_HOLD_MASKS.to(device)
        
        # Create networks with configurable architecture
        self.policy_net = PolicyNetwork(
            state_dim=self.state_dim,
            Z=Z,
            hidden_dim=hidden_dim, 
            activation=activation,
            num_shared_layers=num_shared_layers,
            num_branch_layers=num_branch_layers
        ).to(device)
        self.value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=hidden_dim, 
            activation=activation
        ).to(device)
        
        # Compile networks for optimization if requested and available
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Use 'reduce-overhead' mode - good balance of compile time and performance
                # 'max-autotune' takes too long for initial compilation
                self.policy_net = torch.compile(self.policy_net, mode='reduce-overhead')
                self.value_net = torch.compile(self.value_net, mode='reduce-overhead')
                print("Networks compiled with torch.compile (reduce-overhead mode)")
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
            state_dim=self.state_dim,
            Z=Z,
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
        Decide which dice to hold using categorical selection from 32 hold patterns.
        
        Args:
            game: MultiYahtzee game instance
            
        Returns:
            Tensor of shape (num_games, 5) with 0/1 values for hold decisions
        """
        # Get state using Multi-Yahtzee's get_state method
        state = game.get_state()
        
        # Get hold logits (32 discrete actions)
        with torch.no_grad():
            # Disable AMP for sampling to avoid float16 precision issues with Categorical
            with torch.amp.autocast('cuda', enabled=False):
                logits = self.policy_net(state, 'hold')  # (batch, 32)
            
            if self.training:
                # Sample from categorical distribution during training
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action_idx = dist.sample()  # (batch,)
            else:
                # Use greedy selection during evaluation
                action_idx = torch.argmax(logits, dim=-1)  # (batch,)
            
            # Convert action index to hold mask using precalculated masks
            hold_mask = self.hold_masks[action_idx]  # (batch, 5)
        
        return hold_mask
    
    def decide_category(self, game):
        """
        Decide which category to score based on current game state.
        
        Args:
            game: MultiYahtzee game instance
            
        Returns:
            Tensor of shape (num_games,) with category index (0 to Z*13-1)
        """
        # Get state using Multi-Yahtzee's get_state method
        state = game.get_state()
        
        # Get category probabilities
        with torch.no_grad():
            # Disable AMP for sampling to avoid float16 precision issues with Categorical
            with torch.amp.autocast('cuda', enabled=False):
                logits = self.policy_net(state, 'category')
            
            if self.training:
                # Sample from categorical distribution during training
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                category = dist.sample()  # (batch,)
            else:
                # Use greedy selection during evaluation
                category = torch.argmax(logits, dim=-1)  # (batch,)
        
        return category
    
    def get_state_from_game(self, game):
        """Extract state tensor from Multi-Yahtzee game instance."""
        # Get the flattened state directly from the game
        return game.get_state()
    
    def compute_returns(self, rewards, dones, gamma=0.99):
        """Compute discounted returns."""
        returns = []
        discounted_reward = torch.zeros(rewards[0].shape[0], device=self.device)
        
        for t in reversed(range(len(rewards))):
            r_t = rewards[t].squeeze(-1) if rewards[t].dim() > 1 else rewards[t]  # Handle both shapes
            nonterminal = 0.0 if dones[t] else 1.0
            discounted_reward = r_t + gamma * nonterminal * discounted_reward
            returns.insert(0, discounted_reward)
        
        return torch.stack(returns)
    
    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE."""
        advantages = []
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            # Handle both 1D and 2D reward tensors
            r_t = rewards[t].squeeze(-1) if rewards[t].dim() > 1 else rewards[t]  # (B,)
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
        T = len(states)  # Total timesteps (Z*13 rounds × 3 decisions)
        B = states[0].shape[0]  # Batch size
        steps_per_episode = self.Z * 13 * 3  # Z*13 rounds, 3 decisions per round
        
        # Stack trajectory data once for efficient manipulation
        S = torch.stack(states)  # (T, B, state_dim)
        R = torch.stack([r.squeeze(-1) if r.dim() > 1 else r for r in rewards])  # (T, B)
        
        # Compute values for all states efficiently in a single batch
        with torch.no_grad():
            S_flat = S.reshape(-1, S.shape[-1])  # (T*B, state_dim)
            V_flat = self.value_net(S_flat).squeeze(-1)  # (T*B,)
            V = V_flat.view(T, B)  # Reshape back to (T, B)
        
        # Mark episode boundaries  
        dones = [(i + 1) % steps_per_episode == 0 for i in range(T)]
        
        # Compute returns and advantages using GAE
        returns = self.compute_returns(rewards, dones, gamma=self.gamma)  # (T, B)
        advantages = self.compute_advantages(rewards, V, dones)  # (T, B)
        
        # Identify hold vs category timesteps
        is_hold = torch.tensor([a[0] == 'hold' for a in actions], device=self.device)
        is_cat = ~is_hold
        
        # Build flattened datasets for holds using tensor indexing (no Python loops)
        if is_hold.any():
            # Use tensor indexing to select hold timesteps
            S_hold = S[is_hold].reshape(-1, S.shape[-1])  # (num_holds*B, state_dim)
            R_hold = returns[is_hold].reshape(-1)  # (num_holds*B,)
            Adv_hold = advantages[is_hold].reshape(-1)  # (num_holds*B,)
            
            # Vectorized conversion from hold masks to action indices using bit arithmetic
            # Precompute bit weights for 5-bit encoding (matches format(i, '05b'))
            bit_weights = torch.tensor([16, 8, 4, 2, 1], dtype=torch.float32, device=self.device)  # MSB to LSB
            
            # Extract hold masks from actions using tensor comprehension
            # First get the hold timestep indices
            hold_actions = [actions[i][1] for i in torch.where(is_hold)[0].tolist()]
            H_hold = torch.cat(hold_actions, dim=0) if hold_actions else torch.empty(0, 5, device=self.device)
            
            # Convert binary masks to action indices via dot product with bit weights
            A_hold = (H_hold @ bit_weights).long()  # (num_holds*B,)
            
            # Normalize advantages per type
            Adv_hold = (Adv_hold - Adv_hold.mean()) / (Adv_hold.std() + 1e-8)
            
            # Precompute old log probs for holds
            with torch.no_grad():
                old_logits_h = self.old_policy_net(S_hold, 'hold')
                old_probs_h = F.softmax(old_logits_h, dim=-1)
                old_dist_h = Categorical(old_probs_h)
                old_logp_h = old_dist_h.log_prob(A_hold)
        
        # Build flattened datasets for categories using tensor indexing (no Python loops)
        if is_cat.any():
            # Use tensor indexing to select category timesteps
            S_cat = S[is_cat].reshape(-1, S.shape[-1])  # (num_cats*B, state_dim)
            R_cat = returns[is_cat].reshape(-1)  # (num_cats*B,)
            Adv_cat = advantages[is_cat].reshape(-1)  # (num_cats*B,)
            
            # Extract category actions from actions list
            cat_actions = [actions[i][1] for i in torch.where(is_cat)[0].tolist()]
            A_cat = torch.cat(cat_actions, dim=0) if cat_actions else torch.empty(0, device=self.device, dtype=torch.long)
            if A_cat.dim() > 1:
                A_cat = A_cat.squeeze(-1)  # Ensure 1D
            
            # Normalize advantages per type
            Adv_cat = (Adv_cat - Adv_cat.mean()) / (Adv_cat.std() + 1e-8)
            
            # Precompute old log probs for categories
            with torch.no_grad():
                old_logits_c = self.old_policy_net(S_cat, 'category')
                old_probs_c = F.softmax(old_logits_c, dim=-1)
                old_dist_c = Categorical(old_probs_c)
                old_logp_c = old_dist_c.log_prob(A_cat)
        
        # Large batch size for better GPU utilization
        mb_size = min(2048, B * 6)  # At least full games worth of data
        
        # Pre-allocate tensors for losses on GPU (avoid Python lists)
        # Maximum possible updates: k_epochs * (hold_batches + cat_batches)
        num_holds = is_hold.sum().item()
        num_cats = is_cat.sum().item()
        max_updates = self.k_epochs * ((num_holds * B // mb_size + 1) + 
                                       (num_cats * B // mb_size + 1))
        policy_losses = torch.zeros(max_updates, device=self.device)
        value_losses = torch.zeros(max_updates, device=self.device)
        update_idx = 0
        
        # PPO epochs
        for _ in range(self.k_epochs):
            
            # Process hold decisions
            if is_hold.any():
                Nh = S_hold.shape[0]
                perm_h = torch.randperm(Nh, device=self.device)
                
                for i in range(0, Nh, mb_size):
                    idx = perm_h[i:min(i + mb_size, Nh)]
                    
                    # Use AMP autocast for forward pass
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        # Policy forward pass
                        logits = self.policy_net(S_hold[idx], 'hold')
                        probs = F.softmax(logits, dim=-1)
                        dist = Categorical(probs)
                        logp = dist.log_prob(A_hold[idx])
                        ent = dist.entropy()
                        
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
                    
                    # Store losses in pre-allocated tensor (no Python list overhead)
                    policy_losses[update_idx] = policy_loss.detach()
                    value_losses[update_idx] = value_loss.detach()
                    update_idx += 1
            
            # Process category decisions
            if is_cat.any():
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
                        logp = dist.log_prob(A_cat[idx])
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
                    
                    # Store losses in pre-allocated tensor (no Python list overhead)
                    policy_losses[update_idx] = policy_loss.detach()
                    value_losses[update_idx] = value_loss.detach()
                    update_idx += 1
        
        # Return average losses (single .item() call at the very end)
        if update_idx > 0:
            # Only average the actual updates (not the pre-allocated zeros)
            avg_policy_loss = policy_losses[:update_idx].mean().item()
            avg_value_loss = value_losses[:update_idx].mean().item()
        else:
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
        
        return avg_policy_loss, avg_value_loss