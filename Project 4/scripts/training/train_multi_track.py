"""
Multi-Track Training Script
Trains an agent sequentially on all three tracks for robust performance.

Usage:
    python train_multi_track.py
"""
import os
from src.run import run
from loguru import logger

# Ensure the current working directory is the project root
project_root = os.path.dirname(os.path.abspath(__file__))
# Handle new directory structure: scripts/training/ -> go up 2 levels
if os.path.basename(project_root) == 'training':
    project_root = os.path.dirname(os.path.dirname(project_root))
elif os.path.basename(project_root) == 'src':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)

logger.info(f"Current working directory: {os.getcwd()}")

# Hyperparameters for multi-track training
multi_track_hparams = {
    'experiment_name': 'multi_track_ppo',
    'total_timesteps': 300000,  # Total across all tracks
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
}

print("="*70)
print("Multi-Track Training")
print("="*70)
print("\nThis will train a PPO agent sequentially on all three tracks:")
print("  1. reInvent2019_wide (A to Z Speedway)")
print("  2. reInvent2019_track (Smile Speedway)")
print("  3. Vegas_track (AWS Summit Raceway)")
print("\nNote: This requires restarting the simulation container between tracks.")
print("Starting training...\n")

# Train on all tracks
run(multi_track_hparams, multi_track=True)

print("\n" + "="*70)
print("Multi-Track Training Complete!")
print("="*70)

