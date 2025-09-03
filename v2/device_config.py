import torch

# Set device - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """Get the device to use for tensors."""
    return device

def device_info():
    """Print information about the device being used."""
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("Using CPU (CUDA not available)")
    return device