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