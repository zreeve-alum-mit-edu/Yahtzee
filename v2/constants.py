import torch

# Upper section scoring values
# Row i contains scores for getting 0-5 of value (i+1)
# E.g., row 0: [0,1,2,3,4,5] for 0-5 ones
#       row 1: [0,2,4,6,8,10] for 0-5 twos
UPPER_SCORING_TENSOR = torch.tensor([
    [0, 1, 2, 3, 4, 5],      # Ones
    [0, 2, 4, 6, 8, 10],     # Twos
    [0, 3, 6, 9, 12, 15],    # Threes
    [0, 4, 8, 12, 16, 20],   # Fours
    [0, 5, 10, 15, 20, 25],  # Fives
    [0, 6, 12, 18, 24, 30]   # Sixes
], dtype=torch.float32)

# Bonus threshold and value
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS_VALUE = 35

# Small straight patterns (4 consecutive dice)
# Each row represents a possible small straight pattern
SMALL_STRAIGHT_PATTERNS = torch.tensor([
    [1, 1, 1, 1, 0, 0],  # 1-2-3-4
    [0, 1, 1, 1, 1, 0],  # 2-3-4-5
    [0, 0, 1, 1, 1, 1],  # 3-4-5-6
], dtype=torch.float32)

# Large straight patterns (5 consecutive dice)
# Each row represents a possible large straight pattern
LARGE_STRAIGHT_PATTERNS = torch.tensor([
    [1, 1, 1, 1, 1, 0],  # 1-2-3-4-5
    [0, 1, 1, 1, 1, 1],  # 2-3-4-5-6
], dtype=torch.float32)

SMALL_STRAIGHT_SCORE = 30
LARGE_STRAIGHT_SCORE = 40

# Dice hold masks for 32 discrete actions (2^5 combinations)
# Each row represents which dice to hold (1) or reroll (0)
# Index 0: reroll all, Index 31: hold all
DICE_HOLD_MASKS = torch.tensor([
    [0, 0, 0, 0, 0],  # 0:  00000 - reroll all
    [0, 0, 0, 0, 1],  # 1:  00001
    [0, 0, 0, 1, 0],  # 2:  00010
    [0, 0, 0, 1, 1],  # 3:  00011
    [0, 0, 1, 0, 0],  # 4:  00100
    [0, 0, 1, 0, 1],  # 5:  00101
    [0, 0, 1, 1, 0],  # 6:  00110
    [0, 0, 1, 1, 1],  # 7:  00111
    [0, 1, 0, 0, 0],  # 8:  01000
    [0, 1, 0, 0, 1],  # 9:  01001
    [0, 1, 0, 1, 0],  # 10: 01010
    [0, 1, 0, 1, 1],  # 11: 01011
    [0, 1, 1, 0, 0],  # 12: 01100
    [0, 1, 1, 0, 1],  # 13: 01101
    [0, 1, 1, 1, 0],  # 14: 01110
    [0, 1, 1, 1, 1],  # 15: 01111
    [1, 0, 0, 0, 0],  # 16: 10000
    [1, 0, 0, 0, 1],  # 17: 10001
    [1, 0, 0, 1, 0],  # 18: 10010
    [1, 0, 0, 1, 1],  # 19: 10011
    [1, 0, 1, 0, 0],  # 20: 10100
    [1, 0, 1, 0, 1],  # 21: 10101
    [1, 0, 1, 1, 0],  # 22: 10110
    [1, 0, 1, 1, 1],  # 23: 10111
    [1, 1, 0, 0, 0],  # 24: 11000
    [1, 1, 0, 0, 1],  # 25: 11001
    [1, 1, 0, 1, 0],  # 26: 11010
    [1, 1, 0, 1, 1],  # 27: 11011
    [1, 1, 1, 0, 0],  # 28: 11100
    [1, 1, 1, 0, 1],  # 29: 11101
    [1, 1, 1, 1, 0],  # 30: 11110
    [1, 1, 1, 1, 1],  # 31: 11111 - hold all
], dtype=torch.float32)