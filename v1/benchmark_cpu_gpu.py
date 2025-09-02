import torch
import time
from train_ppo import train_ppo_agent
from device_config import device

print("=" * 60)
print("CPU vs GPU Performance Benchmark")
print("=" * 60)

# Force CPU mode
print("\n1. Testing on CPU...")
print("-" * 40)
original_device = device

# Monkey patch to force CPU
import device_config
device_config.device = torch.device('cpu')

# Reload modules to use CPU
import importlib
import simple_yahtzee
import random_player
import ppo_player
import game_runner
import constants

# Reload all modules with CPU device
importlib.reload(constants)
importlib.reload(simple_yahtzee)
importlib.reload(random_player)
importlib.reload(ppo_player)
importlib.reload(game_runner)

# Run CPU benchmark
start_time = time.time()
print("Starting CPU training...")
train_ppo_agent(num_episodes=10, num_parallel_games=100)
cpu_time = time.time() - start_time
print(f"\nCPU Time: {cpu_time:.2f} seconds")

# Force GPU mode
print("\n2. Testing on GPU...")
print("-" * 40)

# Check if CUDA is available
if torch.cuda.is_available():
    # Monkey patch to force GPU
    device_config.device = torch.device('cuda')
    
    # Reload all modules with GPU device
    importlib.reload(constants)
    importlib.reload(simple_yahtzee)
    importlib.reload(random_player)
    importlib.reload(ppo_player)
    importlib.reload(game_runner)
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Run GPU benchmark
    start_time = time.time()
    print("Starting GPU training...")
    train_ppo_agent(num_episodes=10, num_parallel_games=100)
    torch.cuda.synchronize()  # Ensure all GPU operations complete
    gpu_time = time.time() - start_time
    print(f"\nGPU Time: {gpu_time:.2f} seconds")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"CPU Time: {cpu_time:.2f} seconds")
    print(f"GPU Time: {gpu_time:.2f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"Time saved: {cpu_time - gpu_time:.2f} seconds")
    
    # Memory usage
    print(f"\nGPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
else:
    print("CUDA not available, cannot run GPU benchmark")

# Restore original device
device_config.device = original_device