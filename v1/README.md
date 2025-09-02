# Yahtzee PPO v1 - Simplified Upper Section Only

## Achievement
Successfully trained a PPO agent to achieve **optimal performance (59 average reward)** in simplified upper-only Yahtzee using pure reinforcement learning.

## Game Rules
- 6 rounds (one per upper category: 1s through 6s)
- 3 rolls per round with hold decisions
- Must score in each category exactly once
- -50 penalty for trying to score in already-used category

## Architecture
- **Network**: Forked architecture
  - 3 shared layers (128 hidden dim)
  - 2 branch layers per head (hold/category)
  - Hold head: 5 independent Bernoulli outputs
  - Category head: 6-way categorical output
- **State**: 39 dimensions (dice:30 + categories:6 + turn:3)
- **Training**: Flattened trajectory processing for efficiency

## Key Hyperparameters
- **Parallel games**: 10,000
- **Minibatch size**: 2,048 (flattened training)
- **Learning rate**: 1e-4 (3x multiplier for hold head)
- **Entropy**: Annealed from 0.05 to 0 over 4,000 episodes
- **K epochs**: 2
- **GAE λ**: 0.95
- **PPO ε-clip**: 0.2

## Results
- **Convergence**: ~2,000 episodes to reach 58.5
- **Final performance**: 59 average reward (theoretical maximum)
- **Comparison**: Matches rule-based optimal play

## Critical Fixes Made
1. Fixed mini-batching bug (was processing all games instead of mini-batch)
2. Fixed double-averaging bug in loss calculation
3. Implemented flattened training (~20x speedup)
4. Added entropy annealing to eliminate residual randomness
5. Used forked architecture for action-specific specialization

## Files
- `simple_yahtzee.py` - Vectorized game implementation
- `ppo_bernoulli.py` - PPO agent with Bernoulli holds
- `train_ppo_bernoulli.py` - Training script
- `game_runner.py` - Episode management
- `basic_player.py` - Rule-based baseline (56.2 avg)
- `random_player.py` - Random baseline (44 avg)