"""
Generate training results summary and visualizations.
"""
import json
import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def extract_tensorboard_metrics(runs_dir="runs"):
    """Extract metrics from TensorBoard logs."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    
    # Get latest run
    run_dirs = sorted(runs_path.glob("*"), key=os.path.getmtime, reverse=True)
    if not run_dirs:
        return None
    
    latest_run = run_dirs[0]
    event_files = list(latest_run.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    
    try:
        ea = EventAccumulator(str(latest_run))
        ea.Reload()
        
        metrics = {}
        for tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(tag)
            metrics[tag] = {
                'values': [s.value for s in scalar_events],
                'steps': [s.step for s in scalar_events],
                'wall_times': [s.wall_time for s in scalar_events]
            }
        
        return {
            'run_name': latest_run.name,
            'metrics': metrics
        }
    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        return None

def generate_training_report():
    """Generate comprehensive training report."""
    
    print("="*70)
    print("TRAINING RESULTS SUMMARY")
    print("="*70)
    
    # Extract TensorBoard metrics
    tb_data = extract_tensorboard_metrics()
    
    # Training statistics from console output
    training_stats = {
        "total_timesteps": 100000,
        "total_episodes": 1870,
        "mean_reward": 26.86,
        "std_reward": 21.01,
        "mean_length": 53.6,
        "std_length": 39.4,
        "best_reward": 162.54,
        "recent_mean_reward": 18.93,
        "evaluation_mean_reward": 25.09,
        "evaluation_std_reward": 6.97,
        "evaluation_mean_length": 40.80,
        "evaluation_std_length": 7.68
    }
    
    # Calculate additional metrics
    training_stats["reward_range"] = {
        "min_expected": training_stats["mean_reward"] - training_stats["std_reward"],
        "max_expected": training_stats["mean_reward"] + training_stats["std_reward"]
    }
    
    training_stats["episodes_per_update"] = training_stats["total_timesteps"] / (training_stats["total_episodes"] / 2048) if training_stats["total_episodes"] > 0 else 0
    training_stats["avg_steps_per_episode"] = training_stats["total_timesteps"] / training_stats["total_episodes"] if training_stats["total_episodes"] > 0 else 0
    
    # Save to JSON
    output_file = "training_results.json"  # Already in results/ directory
    with open(output_file, 'w') as f:
        json.dump({
            "training_stats": training_stats,
            "tensorboard_run": tb_data["run_name"] if tb_data else None,
            "available_metrics": list(tb_data["metrics"].keys()) if tb_data else []
        }, f, indent=2)
    
    print(f"\n[OK] Training results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\nTotal Training:")
    print(f"  Timesteps: {training_stats['total_timesteps']:,}")
    print(f"  Episodes: {training_stats['total_episodes']:,}")
    print(f"  Avg Steps/Episode: {training_stats['avg_steps_per_episode']:.1f}")
    
    print(f"\nReward Statistics:")
    print(f"  Mean Reward: {training_stats['mean_reward']:.2f} ± {training_stats['std_reward']:.2f}")
    print(f"  Best Reward: {training_stats['best_reward']:.2f}")
    print(f"  Recent (last 10) Mean: {training_stats['recent_mean_reward']:.2f}")
    print(f"  Expected Range: [{training_stats['reward_range']['min_expected']:.2f}, {training_stats['reward_range']['max_expected']:.2f}]")
    
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean Length: {training_stats['mean_length']:.1f} ± {training_stats['std_length']:.1f} steps")
    print(f"  Range: ~{training_stats['mean_length'] - training_stats['std_length']:.1f} to ~{training_stats['mean_length'] + training_stats['std_length']:.1f} steps")
    
    print(f"\nEvaluation Results (5 episodes):")
    print(f"  Mean Reward: {training_stats['evaluation_mean_reward']:.2f} ± {training_stats['evaluation_std_reward']:.2f}")
    print(f"  Mean Length: {training_stats['evaluation_mean_length']:.1f} ± {training_stats['evaluation_std_length']:.1f} steps")
    
    # Analysis
    print(f"\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    if training_stats['best_reward'] > training_stats['mean_reward'] * 2:
        print("\n[NOTE] Large gap between best and mean reward suggests:")
        print("  - Inconsistent performance (some episodes much better than others)")
        print("  - Agent may need more training for stability")
        print("  - Consider reward function tuning")
    else:
        print("\n[OK] Best reward is reasonably close to mean (consistent performance)")
    
    if training_stats['recent_mean_reward'] < training_stats['mean_reward']:
        print("\n[WARNING] Recent performance lower than overall mean")
        print("  - Agent may have regressed or hit a plateau")
        print("  - Consider: more training, learning rate adjustment, or reward tuning")
    else:
        print("\n[OK] Recent performance is improving or stable")
    
    if training_stats['mean_length'] < 30:
        print("\n[WARNING] Short episode lengths suggest agent crashes early")
        print("  - May need better reward function")
        print("  - Consider: reward for staying on track, penalizing crashes")
    else:
        print(f"\n[OK] Episode lengths are reasonable ({training_stats['mean_length']:.1f} steps)")
        print("  - Agent is staying on track for meaningful periods")
    
    print(f"\n" + "="*70)
    print("MODEL FILES")
    print("="*70)
    
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.zip"))
        if model_files:
            print(f"\nFound {len(model_files)} model file(s):")
            for m in sorted(model_files, key=os.path.getmtime, reverse=True)[:5]:
                size_mb = m.stat().st_size / (1024 * 1024)
                print(f"  - {m.name} ({size_mb:.2f} MB)")
        else:
            print("\nNo model files found")
    else:
        print("\nModels directory not found")
    
    print(f"\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    print("\nTo view training plots, run:")
    print("  tensorboard --logdir runs")
    print("\nThen open http://localhost:6006 in your browser")
    
    if tb_data:
        print(f"\nAvailable TensorBoard metrics:")
        for metric in tb_data['metrics'].keys():
            print(f"  - {metric}")
    
    print("\n" + "="*70)
    
    return training_stats

if __name__ == "__main__":
    stats = generate_training_report()

