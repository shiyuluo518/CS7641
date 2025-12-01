"""
Plot real training metrics from saved training data.
This uses the actual training metrics, not synthetic data.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def load_training_metrics(metrics_file='results/training_metrics.json'):
    """Load training metrics from saved file."""
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        print(f"[ERROR] Training metrics file not found: {metrics_file}")
        print("Training metrics should be saved during training.")
        return None
    
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    return data

def plot_training_metrics_from_data(training_data, output_dir="results/plots"):
    """Plot the 3 training metrics from real training data."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    rewards = np.array(training_data['episode_rewards'])
    lengths = np.array(training_data['episode_lengths'])
    progress = np.array(training_data['episode_progress'])
    
    # Calculate steps (approximate if not available)
    if 'episode_steps' in training_data and len(training_data['episode_steps']) == len(rewards):
        steps = np.array(training_data['episode_steps'])
    else:
        # Estimate steps: assume episodes are evenly distributed
        n_episodes = len(rewards)
        steps = np.linspace(0, 200000, n_episodes)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Training Metrics - Time Trial', fontsize=16, fontweight='bold')
    
    window = min(50, len(rewards) // 10) if len(rewards) > 10 else 10
    
    # 1. Episode Reward
    axes[0].plot(steps, rewards, alpha=0.6, linewidth=1, color='blue', label='Episode Reward')
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_steps = steps[window-1:]
        axes[0].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Episode Reward', fontsize=12)
    axes[0].set_title('1. Episode Reward Over Training', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Progress Percentage
    axes[1].plot(steps, progress, alpha=0.6, linewidth=1, color='green', label='Progress %')
    if len(progress) > window:
        moving_avg = np.convolve(progress, np.ones(window)/window, mode='valid')
        moving_steps = steps[window-1:]
        axes[1].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Target')
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Progress (%)', fontsize=12)
    axes[1].set_title('2. Progress Percentage Over Training', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([0, 110])
    
    # 3. Lap Time Proxy (Episode Length)
    axes[2].plot(steps, lengths, alpha=0.6, linewidth=1, color='orange', label='Episode Length (Lap Time Proxy)')
    if len(lengths) > window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        moving_steps = steps[window-1:]
        axes[2].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
    axes[2].set_xlabel('Training Step', fontsize=12)
    axes[2].set_ylabel('Episode Length (Lap Time Proxy)', fontsize=12)
    axes[2].set_title('3. Lap Time Proxy Over Training (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_metrics_three_plots.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {output_path / 'training_metrics_three_plots.png'}")
    plt.close()
    
    # Print statistics
    print(f"\nTraining Statistics:")
    print(f"  Total Episodes: {len(rewards)}")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean Progress: {np.mean(progress):.1f}% ± {np.std(progress):.1f}%")
    print(f"  Best Progress: {np.max(progress):.1f}%")
    print(f"  Recent (last 10) Mean Progress: {np.mean(progress[-10:]):.1f}%")

if __name__ == "__main__":
    print("="*70)
    print("Plotting Real Training Metrics")
    print("="*70)
    
    training_data = load_training_metrics()
    if training_data:
        plot_training_metrics_from_data(training_data)
        print("\n" + "="*70)
        print("Plots generated successfully!")
        print("="*70)
    else:
        print("\n[ERROR] Could not load training metrics.")
        print("Please run training first to generate training_metrics.json")

