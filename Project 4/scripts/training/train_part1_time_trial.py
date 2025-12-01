"""
Part I: Time-Trial Training Script
Trains an agent for Time-Trial racing (no obstacles, no bots).

Usage:
    python train_part1_time_trial.py
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

# Hyperparameters for Time-Trial - Optimized for better convergence
time_trial_hparams = {
    'experiment_name': 'time_trial_ppo',
    'total_timesteps': 400000,  # Significantly increased for better convergence
    'learning_rate': 2.5e-4,    # Optimized learning rate for stability
    'n_steps': 2048,
    'batch_size': 128,           # Increased batch size for more stable updates
    'n_epochs': 10,
    'gamma': 0.99,
}

print("="*70)
print("Part I: Time-Trial Training")
print("="*70)
print("\nThis will train a PPO agent for Time-Trial racing.")
print("The agent will learn to:")
print("  - Stay on track")
print("  - Follow the center line")
print("  - Make progress quickly")
print("  - Complete laps in minimum time")
print("\nStarting training...\n")

# Train on single track first (can enable multi-track later)
run(time_trial_hparams, multi_track=False)

print("\n" + "="*70)
print("Time-Trial Training Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Check evaluation results in evaluations/")
print("  2. View demo video in demos/")
print("  3. Review TensorBoard logs: tensorboard --logdir runs")
print("  4. If performance is good, proceed to Part II")

