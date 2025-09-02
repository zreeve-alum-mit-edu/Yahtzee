# Yahtzee PPO Agent Project Summary

## Project Overview
Built a parallel Yahtzee game engine with PPO (Proximal Policy Optimization) reinforcement learning agent, entirely using PyTorch tensors for GPU acceleration.

## Architecture Decisions

### 1. Tensor-Based Game State
- **Decision**: All game state represented as tensors (no Python objects/lists)
- **Why**: Enables massive parallelization on GPU, no loops needed
- **Implementation**:
  - Dice: `(num_games, 5, 6)` - one-hot encoded
  - Scores: `(num_games, 6, 7)` - one-hot for counts 0-5 + unscored flag
  - Turn: `(num_games, 3)` - one-hot for game phase
  - Penalties: `(num_games, 1)` - accumulated penalty count

### 2. Dice Sorting
- **Decision**: Always sort dice in ascending order after rolling
- **Why**: Achieves permutation invariance without complex architectures
- **Impact**: Hold patterns become semantic (e.g., pattern 3 always means "hold two lowest")

### 3. Hold Action Space
- **Decision**: 32-way softmax instead of 5 independent binary decisions
- **Why**: Captures dependencies between dice (holding patterns are correlated)
- **Implementation**: Lookup table for pattern index ↔ binary mask conversion (stays on GPU)

### 4. Separate Action Heads
- **Decision**: Different neural network heads for hold vs category decisions
- **Why**: Fundamentally different action types and contexts
- **Implementation**: Shared backbone, specialized output heads

### 5. Penalty System
- **Decision**: -10 reward for choosing already-scored categories
- **Why**: Teaches agent valid play through negative reinforcement
- **Implementation**: Random unscored category selected when invalid choice made

### 6. GPU-First Design
- **Decision**: All tensors created and stay on GPU
- **Why**: 3.29x speedup demonstrated, scales better with batch size
- **Implementation**: Even lookup tables and constants on GPU

## Key Design Principles (Per User Instructions)

1. **No Imposed Strategy**: User emphasized not imposing preconceptions about what's relevant. Example: Rejected value-based hold actions in favor of letting agent discover patterns.

2. **Start Simple**: User consistently requested starting with minimal implementation first, then adding complexity.

3. **Tensor Operations Only**: Avoid loops, use vectorized operations throughout.

4. **Test Everything**: Comprehensive test suite for each component before moving on.

## Project Structure

```
/mnt/c/GIT/Yahtzee/
├── simple_yahtzee.py      # Core game engine
├── game_runner.py         # Game orchestration & SAR tracking
├── random_player.py       # Baseline random agent
├── ppo_player.py         # PPO neural network agent
├── train_ppo.py          # Training loop
├── device_config.py      # GPU/CPU device management
├── constants.py          # Lookup tables (hold patterns)
└── tests/
    ├── test_dice_rolling.py
    ├── test_score_category.py
    ├── test_penalties.py
    ├── test_dice_sorting.py
    └── test_scoring_consistency.py
```

## Current State

### What Works
- ✅ Parallel game execution (100+ games simultaneously)
- ✅ Complete upper Yahtzee scoring
- ✅ PPO training with proper SAR tracking
- ✅ GPU acceleration throughout
- ✅ Penalty system for invalid moves
- ✅ Comprehensive test coverage

### Performance
- GPU provides 3.29x speedup over CPU
- Can run 100 parallel games efficiently
- PPO shows improvement over random baseline (~5 point advantage)

### Known Limitations
- Only upper Yahtzee implemented (no lower section)
- PPO needs more training episodes for optimal play
- Still gets ~2.5 penalties per game (room for improvement)

## Next Steps Potential
- Add lower Yahtzee scoring (straights, full house, etc.)
- Implement bonus for upper section (35 points for 63+)
- Tune PPO hyperparameters
- Add reward shaping for better learning
- Implement curriculum learning (start with fewer categories)

## Important Technical Notes

1. **Sorting Behavior**: Dice are sorted AFTER applying hold mask and rolling. Held values persist but may change position.

2. **Reward vs Score**: Total rewards ≠ traditional score. Rewards include -10 penalties for training signal.

3. **State Representation**: 76-dimensional flat vector for neural network (30 dice + 42 scores + 3 turn + 1 penalty).

4. **Hold Patterns**: Binary patterns map to indices (e.g., [1,0,1,0,0] = index 5 = binary 00101).

5. **One-Hot Scoring**: Score of 3 threes = [0,0,0,1,0,0,0] where index represents count.

## Key Insights

1. **Parallel Processing Power**: Tensor operations enable massive parallelization impossible with traditional OOP approach.

2. **Permutation Invariance**: Sorting is simpler than complex architectures and works perfectly for Yahtzee.

3. **Sparse Rewards Work**: Even with rewards only on scoring (not holding), PPO learns.

4. **GPU Memory Efficient**: Only 34MB for 100 parallel games with neural networks.

## User Preferences to Remember

- Prefers concise responses without preamble
- Wants agent to discover strategies, not have them imposed
- Values starting simple and building up
- Emphasizes testing before moving forward
- Interested in pure tensor operations for efficiency

---
*Last Updated: End of initial development phase*
*Status: Functional PPO agent training successfully on GPU*