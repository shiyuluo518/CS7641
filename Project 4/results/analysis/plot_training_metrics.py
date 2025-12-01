"""
Plot the 3 key training metrics as required by the project:
1. Episode Reward
2. Progress Percentage  
3. Lap Time Proxy (Episode Length)

Usage (from project root):
    python results/plot_training_metrics.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Add project root to path if running from results/ directory
if Path.cwd().name == 'results':
    sys.path.insert(0, str(Path.cwd().parent))

def plot_three_metrics(runs_dir="results/runs", output_dir="results/plots"):
    """
    Plot the 3 required metrics from TensorBoard logs.
    """
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    if not runs_path.exists():
        print(f"[ERROR] Runs directory not found: {runs_dir}")
        return
    
    # Get latest run
    run_dirs = sorted(runs_path.glob("*"), key=os.path.getmtime, reverse=True)
    if not run_dirs:
        print("[ERROR] No training runs found")
        return
    
    latest_run = run_dirs[0]
    print(f"Loading metrics from: {latest_run.name}")
    
    try:
        ea = EventAccumulator(str(latest_run))
        ea.Reload()
        
        # Get available metrics
        available_metrics = ea.Tags()['scalars']
        print(f"Available metrics: {available_metrics}")
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('Training Metrics - Time Trial', fontsize=16, fontweight='bold')
        
        # 1. Episode Reward
        if 'train/episode_reward' in available_metrics:
            rewards = ea.Scalars('train/episode_reward')
            steps = [s.step for s in rewards]
            values = [s.value for s in rewards]
            
            axes[0].plot(steps, values, alpha=0.6, linewidth=1, color='blue', label='Episode Reward')
            
            # Add moving average
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                moving_steps = steps[window-1:]
                axes[0].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            
            axes[0].set_xlabel('Training Step', fontsize=12)
            axes[0].set_ylabel('Episode Reward', fontsize=12)
            axes[0].set_title('1. Episode Reward Over Training', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('1. Episode Reward (Not Available)', fontsize=14)
        
        # 2. Progress Percentage
        if 'train/episode_progress' in available_metrics:
            progress = ea.Scalars('train/episode_progress')
            steps = [s.step for s in progress]
            values = [s.value for s in progress]
            
            axes[1].plot(steps, values, alpha=0.6, linewidth=1, color='green', label='Progress %')
            
            # Add moving average
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                moving_steps = steps[window-1:]
                axes[1].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            
            # Add 100% line
            axes[1].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% Target')
            
            axes[1].set_xlabel('Training Step', fontsize=12)
            axes[1].set_ylabel('Progress (%)', fontsize=12)
            axes[1].set_title('2. Progress Percentage Over Training', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_ylim([0, 110])
        else:
            axes[1].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('2. Progress Percentage (Not Available)', fontsize=14)
        
        # 3. Lap Time Proxy (Episode Length)
        if 'train/lap_time_proxy' in available_metrics:
            lap_times = ea.Scalars('train/lap_time_proxy')
            steps = [s.step for s in lap_times]
            values = [s.value for s in lap_times]
            
            axes[2].plot(steps, values, alpha=0.6, linewidth=1, color='orange', label='Lap Time Proxy (Steps)')
            
            # Add moving average
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                moving_steps = steps[window-1:]
                axes[2].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            
            axes[2].set_xlabel('Training Step', fontsize=12)
            axes[2].set_ylabel('Lap Time Proxy (Episode Length)', fontsize=12)
            axes[2].set_title('3. Lap Time Proxy Over Training (Lower is Better)', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
        elif 'train/episode_length' in available_metrics:
            # Fallback to episode_length if lap_time_proxy not available
            lengths = ea.Scalars('train/episode_length')
            steps = [s.step for s in lengths]
            values = [s.value for s in lengths]
            
            axes[2].plot(steps, values, alpha=0.6, linewidth=1, color='orange', label='Episode Length (Lap Time Proxy)')
            
            # Add moving average
            if len(values) > 10:
                window = min(50, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                moving_steps = steps[window-1:]
                axes[2].plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window})')
            
            axes[2].set_xlabel('Training Step', fontsize=12)
            axes[2].set_ylabel('Episode Length (Lap Time Proxy)', fontsize=12)
            axes[2].set_title('3. Lap Time Proxy Over Training (Lower is Better)', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
        else:
            axes[2].text(0.5, 0.5, 'Metric not available', ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('3. Lap Time Proxy (Not Available)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path / 'training_metrics_three_plots.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_path / 'training_metrics_three_plots.png'}")
        plt.close()
        
        print(f"\n[SUCCESS] Three metrics plots saved to {output_path}/")
        
    except Exception as e:
        print(f"[ERROR] Failed to extract metrics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_three_metrics()
