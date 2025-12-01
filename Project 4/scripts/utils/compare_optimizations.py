"""
Comparison tool for baseline vs optimized reward functions and hyperparameters.

Usage:
    python compare_optimizations.py [--task time_trial|obstacle_avoidance|head_to_bot]
"""
import argparse
import yaml
from pathlib import Path

def compare_reward_functions(task='time_trial'):
    """Compare baseline vs optimized reward functions."""
    
    baseline_files = {
        'time_trial': 'configs/rewards/reward_function.py',
        'obstacle_avoidance': 'configs/rewards/reward_function_obstacle_avoidance.py',
        'head_to_bot': 'configs/rewards/reward_function_head_to_bot.py',
    }
    
    optimized_files = {
        'time_trial': 'configs/rewards/reward_function_optimized.py',
        'obstacle_avoidance': 'configs/rewards/reward_function_obstacle_avoidance_optimized.py',
        'head_to_bot': 'configs/rewards/reward_function_head_to_bot_optimized.py',
    }
    
    baseline_file = Path(baseline_files[task])
    optimized_file = Path(optimized_files[task])
    
    print(f"\n{'='*70}")
    print(f"Reward Function Comparison: {task.replace('_', ' ').title()}")
    print(f"{'='*70}\n")
    
    if not baseline_file.exists():
        print(f"⚠️  Baseline file not found: {baseline_file}")
        return
    
    if not optimized_file.exists():
        print(f"⚠️  Optimized file not found: {optimized_file}")
        return
    
    # Read and compare key features
    baseline_content = baseline_file.read_text()
    optimized_content = optimized_file.read_text()
    
    print("Key Improvements in Optimized Version:\n")
    
    improvements = {
        'Waypoint-based progress': 'closest_waypoints' in optimized_content and 'closest_waypoints' not in baseline_content,
        'Steering smoothness': 'steering_penalty' in optimized_content and 'steering_penalty' not in baseline_content,
        'Proper speed normalization': 'MAX_SPEED' in optimized_content,
        'Heading alignment': 'heading_reward' in optimized_content or ('heading' in optimized_content and 'heading_reward' not in baseline_content),
        'Dynamic thresholds': 'CRITICAL_DISTANCE' in optimized_content or 'WARNING_DISTANCE' in optimized_content,
    }
    
    for improvement, present in improvements.items():
        status = "✅" if present else "❌"
        print(f"  {status} {improvement}")
    
    print(f"\nBaseline: {baseline_file}")
    print(f"Optimized: {optimized_file}\n")


def compare_hyperparameters(task='time_trial'):
    """Compare baseline vs optimized hyperparameters."""
    
    baseline_file = Path('configs/hyper_params.yaml')
    
    optimized_files = {
        'time_trial': 'configs/hyperparams/hyper_params_optimized_time_trial.yaml',
        'obstacle_avoidance': 'configs/hyperparams/hyper_params_optimized_obstacle_avoidance.yaml',
        'head_to_bot': 'configs/hyperparams/hyper_params_optimized_head_to_bot.yaml',
    }
    
    optimized_file = Path(optimized_files[task])
    
    print(f"\n{'='*70}")
    print(f"Hyperparameter Comparison: {task.replace('_', ' ').title()}")
    print(f"{'='*70}\n")
    
    if not baseline_file.exists():
        print(f"⚠️  Baseline file not found: {baseline_file}")
        return
    
    if not optimized_file.exists():
        print(f"⚠️  Optimized file not found: {optimized_file}")
        return
    
    # Load YAML files
    with open(baseline_file) as f:
        baseline = yaml.safe_load(f)
    
    with open(optimized_file) as f:
        optimized = yaml.safe_load(f)
    
    # Compare key hyperparameters
    key_params = [
        'learning_rate',
        'batch_size',
        'ent_coef',
        'gamma',
        'gae_lambda',
        'n_steps',
        'n_epochs',
    ]
    
    print("Hyperparameter Changes:\n")
    print(f"{'Parameter':<20} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 70)
    
    for param in key_params:
        baseline_val = baseline.get(param, 'N/A')
        optimized_val = optimized.get(param, 'N/A')
        
        if baseline_val != 'N/A' and optimized_val != 'N/A':
            if baseline_val != optimized_val:
                change = f"{baseline_val} → {optimized_val}"
                print(f"{param:<20} {str(baseline_val):<15} {str(optimized_val):<15} {change:<15}")
            else:
                print(f"{param:<20} {str(baseline_val):<15} {str(optimized_val):<15} {'(unchanged)':<15}")
        else:
            print(f"{param:<20} {str(baseline_val):<15} {str(optimized_val):<15} {'N/A':<15}")
    
    print(f"\nBaseline: {baseline_file}")
    print(f"Optimized: {optimized_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs optimized configurations')
    parser.add_argument('--task', type=str, default='time_trial',
                        choices=['time_trial', 'obstacle_avoidance', 'head_to_bot'],
                        help='Task to compare')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Optimization Comparison Tool")
    print("="*70)
    
    compare_reward_functions(args.task)
    compare_hyperparameters(args.task)
    
    print("\n" + "="*70)
    print("To use optimized versions:")
    print("="*70)
    print(f"\n1. Copy optimized reward function:")
    print(f"   cp configs/rewards/reward_function_{args.task}_optimized.py configs/rewards/reward_function.py")
    print(f"\n2. Copy optimized hyperparameters:")
    print(f"   cp configs/hyperparams/hyper_params_optimized_{args.task}.yaml configs/hyper_params.yaml")
    print(f"\n3. Train with optimized configuration")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

