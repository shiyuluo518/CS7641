"""
Iteration 1: Baseline Training
Establishes baseline using conservative hyperparameters on the simplest track.

According to optimization plan:
- Duration: 4-6 hours training time
- Success metric: consistent track completion
- Track: reInvent2019_wide (simplest track)
- Hyperparameters: Conservative baseline (batch_size=64, learning_rate=3e-4, ent_coef=0.01)

Usage:
    python train_iteration1_baseline.py
"""
import os
import subprocess
import sys
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

# Check if DeepRacer container is running
def check_deepracer_container():
    """Check if DeepRacer container is running."""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=deepracer', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            check=False
        )
        return 'deepracer' in result.stdout
    except Exception as e:
        logger.warning(f"Could not check Docker container status: {e}")
        return False

# Ensure DeepRacer container is running
if not check_deepracer_container():
    logger.error("DeepRacer container is not running!")
    logger.info("Please start the DeepRacer container first:")
    logger.info("  Windows: powershell -ExecutionPolicy Bypass -File scripts/start_deepracer.ps1")
    logger.info("  Linux/Mac: source scripts/start_deepracer.sh")
    logger.info("Or use the helper script: powershell -ExecutionPolicy Bypass -File scripts/ensure_deepracer_running.ps1")
    sys.exit(1)

logger.info("DeepRacer container is running. Proceeding with training...")

# Iteration 1: Baseline Conservative Hyperparameters
iteration1_hparams = {
    'experiment_name': 'time_trial_ppo_iteration1_baseline',
    'total_timesteps': 200000,  # Baseline training duration
    'learning_rate': 3e-4,      # Standard PPO learning rate
    'n_steps': 2048,            # Experience buffer size
    'batch_size': 64,           # Conservative batch size
    'n_epochs': 10,             # Optimal policy refinement
    'gamma': 0.99,              # Discount factor for racing
    'ent_coef': 0.01,           # Initial exploration coefficient
}

print("="*70)
print("Iteration 1: Baseline Training")
print("="*70)
print("\nObjective: Establish baseline with conservative hyperparameters")
print("Track: reInvent2019_wide (simplest track)")
print("\nHyperparameters:")
print(f"  - Learning Rate: {iteration1_hparams['learning_rate']}")
print(f"  - Batch Size: {iteration1_hparams['batch_size']}")
print(f"  - Entropy Coefficient: {iteration1_hparams['ent_coef']}")
print(f"  - Total Timesteps: {iteration1_hparams['total_timesteps']:,}")
print("\nSuccess Metric: Consistent track completion")
print("Expected Duration: 4-6 hours")
print("\nStarting training...\n")

# Train on single track (simplest track)
run(iteration1_hparams, multi_track=False)

print("\n" + "="*70)
print("Iteration 1: Baseline Training Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Check evaluation results in results/evaluations/")
print("  2. Review TensorBoard logs: tensorboard --logdir results/runs")
print("  3. Verify consistent track completion (target: >90% completion rate)")
print("  4. If baseline is successful, proceed to Iteration 2")
print("  5. Run: python train_iteration2_optimized.py")


