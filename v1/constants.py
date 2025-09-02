import torch
from device_config import device

# Create hold pattern lookup table
# Maps pattern index (0-31) to binary mask (5 dice)
def create_hold_pattern_lookup():
    """Create a lookup table for all 32 possible hold patterns."""
    lookup = torch.zeros(32, 5, device=device)
    for pattern_idx in range(32):
        for die_idx in range(5):
            if pattern_idx & (1 << die_idx):
                lookup[pattern_idx, die_idx] = 1
    return lookup

# Create the constant lookup table
HOLD_PATTERN_LOOKUP = create_hold_pattern_lookup()

# Reverse lookup - for each possible mask, what's its index?
# This is trickier since we need to handle batches, so we'll keep the function for now
def mask_to_index_batch(masks):
    """
    Convert batch of binary masks to pattern indices.
    
    Args:
        masks: Tensor of shape (batch_size, 5) with binary values
        
    Returns:
        Tensor of shape (batch_size,) with pattern indices (0-31)
    """
    # Convert binary mask to integer by treating as binary number
    # [1,0,1,0,0] -> 1*1 + 0*2 + 1*4 + 0*8 + 0*16 = 5
    powers = torch.tensor([1, 2, 4, 8, 16], device=device, dtype=torch.float32)
    indices = (masks * powers).sum(dim=1).long()
    return indices