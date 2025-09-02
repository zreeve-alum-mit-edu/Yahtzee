# Yahtzee PPO - Reinforcement Learning Implementation

This repository contains progressive implementations of Yahtzee using Proximal Policy Optimization (PPO).

## Project Structure

### v1/ - Simplified Upper Section Only ✅
**Status**: Complete - Achieved optimal performance (59 average reward)

A simplified version focusing only on the upper section (1s through 6s) to prove the PPO approach works. Successfully learned optimal hold and category selection strategies from scratch.

Key achievements:
- Optimal performance matching theoretical maximum
- Efficient flattened training implementation
- ~20x training speedup through architectural improvements
- Clean separation of hold decisions (5 Bernoulli) and category selection (6-way categorical)

### v2/ - Full Game Implementation 🚧
**Status**: In Development

Full Yahtzee implementation with all 13 categories, bonus rules, and complete scoring system.

## Technical Highlights

- **Fully vectorized game engine** - Processes 10,000+ games in parallel on GPU
- **Flattened PPO training** - Treats trajectory as supervised learning problem for efficiency  
- **Forked network architecture** - Separate branches for different action types
- **Entropy annealing** - Smooth transition from exploration to exploitation
- **GAE advantages** - Proper credit assignment with episode boundaries

## Performance

| Version | Game Variant | Training Episodes | Average Score | Status |
|---------|-------------|------------------|---------------|---------|
| v1 | Upper only (6 categories) | ~2,000 | 59 (optimal) | ✅ Complete |
| v2 | Full game (13 categories) | TBD | TBD | 🚧 In progress |

## Requirements

- PyTorch with CUDA support
- Python 3.8+
- GPU with 8GB+ VRAM recommended for parallel training

## Citation

If you use this code for research, please cite:
```
@software{yahtzee_ppo,
  title={Yahtzee PPO: Efficient Reinforcement Learning for Dice Games},
  author={[Your name]},
  year={2024},
  url={https://github.com/yourusername/yahtzee}
}
```