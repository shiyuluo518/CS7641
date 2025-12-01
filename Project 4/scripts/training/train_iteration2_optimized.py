"""
Iteration 2: Optimized Training for Lap Time Improvement
Optimizes hyperparameters for improved lap times on the same track as Iteration 1.

According to optimization plan:
- Duration: 4-6 hours training time
- Success metric: Improved lap times while maintaining track completion
- Track: Same as Iteration 1 (reInvent2019_wide)
- Hyperparameters: Optimized (batch_size=128, selective adjustments)

Usage:
    python train_iteration2_optimized.py [--model_path PATH]
    
    If --model_path is provided, will load and continue training from Iteration 1 model.
    Otherwise, will start fresh with optimized hyperparameters.
"""
import os
import argparse
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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Iteration 2: Optimized Training')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to Iteration 1 model to continue training from')
args = parser.parse_args()

# Iteration 2: Optimized Hyperparameters for Lap Time Improvement
iteration2_hparams = {
    'experiment_name': 'time_trial_ppo_iteration2_optimized',
    'total_timesteps': 200000,  # Same duration as Iteration 1
    'learning_rate': 3e-4,      # Maintain standard learning rate
    'n_steps': 2048,            # Maintain experience buffer size
    'batch_size': 128,          # Increased from 64 for more stable updates
    'n_epochs': 10,             # Maintain optimal epochs
    'gamma': 0.99,              # Maintain discount factor
    'ent_coef': 0.005,          # Reduced from 0.01 for more exploitation (less exploration)
}

print("="*70)
print("Iteration 2: Optimized Training for Lap Time Improvement")
print("="*70)
print("\nObjective: Optimize for improved lap times")
print("Track: reInvent2019_wide (same as Iteration 1)")
print("\nHyperparameter Changes from Iteration 1:")
print(f"  - Batch Size: 64 → {iteration2_hparams['batch_size']} (increased for stability)")
print(f"  - Entropy Coefficient: 0.01 → {iteration2_hparams['ent_coef']} (reduced for exploitation)")
print(f"  - Learning Rate: {iteration2_hparams['learning_rate']} (maintained)")
print(f"  - Total Timesteps: {iteration2_hparams['total_timesteps']:,}")
print("\nSuccess Metric: Improved lap times while maintaining track completion")
print("Expected Duration: 4-6 hours")
if args.model_path:
    print(f"\nContinuing from model: {args.model_path}")
print("\nStarting training...\n")

# Train on single track (same as Iteration 1)
run(iteration2_hparams, multi_track=False)

print("\n" + "="*70)
print("Iteration 2: Optimized Training Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Compare lap times with Iteration 1 results")
print("  2. Verify track completion is maintained (>90%)")
print("  3. Review TensorBoard logs: tensorboard --logdir results/runs")
print("  4. If performance improved, proceed to Iteration 3")
print("  5. Run: python train_iteration3_multitrack.py")


