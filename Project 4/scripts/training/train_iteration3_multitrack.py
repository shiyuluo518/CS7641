"""
Iteration 3: Multi-Track Generalization Training
Focuses on generalization through sequential training across all three tracks.

According to optimization plan:
- Duration: 8-12 hours training time
- Success metric: Consistent performance across all tracks
- Tracks: All three tracks (reInvent2019_wide, reInvent2019_track, Vegas_track)
- Hyperparameters: Fine-tuning (learning_rate=1e-4, ent_coef=0.005)
- Initialization: Uses optimized model from Iteration 2

Usage:
    python train_iteration3_multitrack.py [--model_path PATH]
    
    If --model_path is provided, will load and continue training from Iteration 2 model.
    Otherwise, will start fresh with fine-tuning hyperparameters.
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
parser = argparse.ArgumentParser(description='Iteration 3: Multi-Track Generalization')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to Iteration 2 model to continue training from')
args = parser.parse_args()

# Iteration 3: Fine-Tuning Hyperparameters for Multi-Track Generalization
iteration3_hparams = {
    'experiment_name': 'time_trial_ppo_iteration3_multitrack',
    'total_timesteps': 300000,  # Total across all tracks (100k per track)
    'learning_rate': 1e-4,      # Reduced for fine-tuning stability
    'n_steps': 2048,            # Maintain experience buffer size
    'batch_size': 128,          # Maintain increased batch size
    'n_epochs': 10,             # Maintain optimal epochs
    'gamma': 0.99,              # Maintain discount factor
    'ent_coef': 0.005,          # Maintain reduced exploration for fine-tuning
}

print("="*70)
print("Iteration 3: Multi-Track Generalization Training")
print("="*70)
print("\nObjective: Generalize performance across all three tracks")
print("Tracks:")
print("  1. reInvent2019_wide (A to Z Speedway)")
print("  2. reInvent2019_track (Smile Speedway)")
print("  3. Vegas_track (AWS Summit Raceway)")
print("\nHyperparameter Changes from Iteration 2:")
print(f"  - Learning Rate: 3e-4 → {iteration3_hparams['learning_rate']} (reduced for fine-tuning)")
print(f"  - Entropy Coefficient: {iteration3_hparams['ent_coef']} (maintained)")
print(f"  - Batch Size: {iteration3_hparams['batch_size']} (maintained)")
print(f"  - Total Timesteps: {iteration3_hparams['total_timesteps']:,} (100k per track)")
print("\nSuccess Metric: Consistent performance across all tracks")
print("Expected Duration: 8-12 hours")
if args.model_path:
    print(f"\nInitializing from model: {args.model_path}")
print("\nNote: This requires restarting the simulation container between tracks.")
print("Starting training...\n")

# Train on all tracks sequentially
run(iteration3_hparams, multi_track=True)

print("\n" + "="*70)
print("Iteration 3: Multi-Track Generalization Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Evaluate performance on all three tracks")
print("  2. Verify consistent performance across tracks")
print("  3. Review TensorBoard logs: tensorboard --logdir results/runs")
print("  4. Check evaluation results in results/evaluations/")
print("  5. If generalization is successful, proceed to Part II (Object-Avoidance/Head-to-Bot)")


