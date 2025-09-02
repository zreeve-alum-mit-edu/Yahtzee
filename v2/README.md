# Yahtzee PPO v2 - Full Game Implementation

## Goals for v2
- [ ] Implement full Yahtzee with all 13 categories
- [ ] Add lower section (3-of-kind, 4-of-kind, full house, small/large straight, Yahtzee, chance)
- [ ] Implement upper section bonus (35 points for 63+ total)
- [ ] Implement Yahtzee bonus rules
- [ ] More complex state representation for additional categories
- [ ] Potentially larger network to handle increased complexity
- [ ] Compare performance with v1 baseline

## Planned Improvements
- Building on v1's successful flattened training
- Maintaining forked architecture
- Consider categorical hold patterns (32-way) instead of 5 Bernoulli
- Explore different reward shaping for strategic category selection

## Target Performance
- TBD - need to establish theoretical maximum for full Yahtzee
- Compare against human expert play (~250-300 average)