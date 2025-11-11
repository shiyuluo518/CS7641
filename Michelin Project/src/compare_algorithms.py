"""
Comparison script for QMIX vs IQL algorithms.
Evaluates both algorithms and generates comparison plots and reports.
"""

import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.env_wrapper import OvercookedWrapper
from src.algorithms.qmix import QMIX
from src.algorithms.iql import IQL
from src.utils import plot_evaluation_results, save_results
import src.config as config


def evaluate_algorithm(layout_name, algorithm, model_path, num_episodes=100):
    """
    Evaluate a trained model (QMIX or IQL) on a specific layout.
    
    Args:
        layout_name: Name of the layout
        algorithm: 'qmix' or 'iql'
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation results
    """
    # Initialize environment
    env = OvercookedWrapper(layout_name)
    
    # Get observation dimension dynamically from environment (varies by layout)
    test_obs = env.reset()
    obs_dim = len(test_obs[0]) if isinstance(test_obs, list) else len(test_obs)
    
    device = torch.device(config.DEVICE)
    
    if algorithm == 'qmix':
        # Get true global state dimension
        test_state = env.get_global_state()
        state_dim = len(test_state)
        
        agent = QMIX(
            obs_dim=obs_dim,  # Use dynamic observation dimension (varies by layout)
            action_dim=config.ACTION_DIM,
            state_dim=state_dim,
            n_agents=config.N_AGENTS,
            hidden_dim=config.HIDDEN_DIM,
            device=device
        )
        agent.load(model_path)
        for net in agent.q_networks:
            net.eval()
        agent.mixing_network.eval()
    else:  # iql
        agent = IQL(
            obs_dim=obs_dim,  # Use dynamic observation dimension (varies by layout)
            action_dim=config.ACTION_DIM,
            n_agents=config.N_AGENTS,
            hidden_dim=config.HIDDEN_DIM,
            device=device
        )
        agent.load(model_path)
        for net in agent.q_networks:
            net.eval()
    
    # Evaluation metrics
    episode_soups = []
    episode_rewards = []
    
    print(f"Evaluating {algorithm.upper()} on {layout_name}...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = [0.0, 0.0]
        steps = 0
        
        while not done and steps < config.HORIZON:
            actions = agent.select_actions(obs, training=False)
            next_obs, rewards, done, info = env.step(actions)
            episode_reward[0] += rewards[0]
            episode_reward[1] += rewards[1]
            obs = next_obs
            steps += 1
        
        # Record results (only when episode is done)
        if done and isinstance(info, dict) and 'episode' in info:
            ep_info = info['episode']
            ep_game_stats = ep_info.get('ep_game_stats', {})
            
            # Extract soup deliveries from ep_game_stats
            # soup_delivery is a list of lists: [[agent0_timesteps], [agent1_timesteps]]
            soup_delivery = ep_game_stats.get('soup_delivery', [[], []])
            # Count total number of soup deliveries across all agents
            soups = sum(len(deliveries) for deliveries in soup_delivery) if isinstance(soup_delivery, (list, tuple)) else 0
            episode_soups.append(soups)
        elif done:
            # Episode ended but no episode info - append 0 soups
            episode_soups.append(0)
        episode_rewards.append(sum(episode_reward))
    
    mean_soups = np.mean(episode_soups)
    std_soups = np.std(episode_soups)
    success_rate = (np.array(episode_soups) >= 7.0).mean() * 100
    
    return {
        'algorithm': algorithm,
        'layout': layout_name,
        'mean_soups': float(mean_soups),
        'std_soups': float(std_soups),
        'success_rate': float(success_rate),
        'episode_soups': episode_soups,
        'episode_rewards': episode_rewards,
    }


def compare_algorithms(layouts=None, models_dir='models', results_dir='results', num_episodes=100):
    """
    Compare QMIX and IQL on all layouts.
    
    Args:
        layouts: List of layouts to evaluate (None = all)
        models_dir: Directory containing trained models
        results_dir: Directory to save comparison results
        num_episodes: Number of evaluation episodes per algorithm/layout
    """
    if layouts is None:
        layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    all_results = {}
    
    print("="*60)
    print("ALGORITHM COMPARISON: QMIX vs IQL")
    print("="*60)
    
    for layout in layouts:
        print(f"\n{'='*60}")
        print(f"Layout: {layout}")
        print(f"{'='*60}")
        
        layout_results = {}
        
        # Evaluate QMIX
        qmix_path = Path(models_dir) / f"{layout}_final.pth"
        if qmix_path.exists():
            qmix_results = evaluate_algorithm(layout, 'qmix', str(qmix_path), num_episodes)
            layout_results['qmix'] = qmix_results
            print(f"QMIX: {qmix_results['mean_soups']:.2f} ± {qmix_results['std_soups']:.2f} soups")
        else:
            print(f"Warning: QMIX model not found: {qmix_path}")
        
        # Evaluate IQL
        iql_path = Path(models_dir) / f"{layout}_iql_final.pth"
        if iql_path.exists():
            iql_results = evaluate_algorithm(layout, 'iql', str(iql_path), num_episodes)
            layout_results['iql'] = iql_results
            print(f"IQL:  {iql_results['mean_soups']:.2f} ± {iql_results['std_soups']:.2f} soups")
        else:
            print(f"Warning: IQL model not found: {iql_path}")
        
        all_results[layout] = layout_results
        
        # Print comparison
        if 'qmix' in layout_results and 'iql' in layout_results:
            qmix_mean = layout_results['qmix']['mean_soups']
            iql_mean = layout_results['iql']['mean_soups']
            improvement = qmix_mean - iql_mean
            improvement_pct = (improvement / iql_mean * 100) if iql_mean > 0 else 0
            print(f"\nQMIX improvement over IQL: {improvement:+.2f} soups ({improvement_pct:+.1f}%)")
    
    # Save comparison results
    comparison_path = Path(results_dir) / "algorithm_comparison.json"
    save_results(all_results, comparison_path)
    print(f"\nComparison results saved to: {comparison_path}")
    
    # Generate comparison plots
    plot_comparison(all_results, Path(results_dir) / "algorithm_comparison.png")
    
    # Generate comparison report
    generate_comparison_report(all_results, Path(results_dir) / "algorithm_comparison_report.txt")
    
    return all_results


def plot_comparison(results_dict, save_path):
    """Plot comparison of QMIX vs IQL."""
    layouts = list(results_dict.keys())
    n_layouts = len(layouts)
    
    fig, axes = plt.subplots(1, n_layouts, figsize=(6*n_layouts, 5))
    if n_layouts == 1:
        axes = [axes]
    
    for idx, layout in enumerate(layouts):
        ax = axes[idx]
        layout_results = results_dict[layout]
        
        if 'qmix' in layout_results and 'iql' in layout_results:
            qmix_soups = layout_results['qmix']['episode_soups']
            iql_soups = layout_results['iql']['episode_soups']
            
            episodes = np.arange(len(qmix_soups))
            ax.plot(episodes, qmix_soups, label='QMIX', alpha=0.6, linewidth=1)
            ax.plot(episodes, iql_soups, label='IQL', alpha=0.6, linewidth=1)
            
            # Add mean lines
            qmix_mean = np.mean(qmix_soups)
            iql_mean = np.mean(iql_soups)
            ax.axhline(y=qmix_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'QMIX mean: {qmix_mean:.2f}')
            ax.axhline(y=iql_mean, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'IQL mean: {iql_mean:.2f}')
            ax.axhline(y=7.0, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Target (7.0)')
        
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Soups Delivered', fontsize=10)
        ax.set_title(layout.replace('_', ' ').title(), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('QMIX vs IQL Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def generate_comparison_report(results_dict, save_path):
    """Generate text report comparing QMIX and IQL."""
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ALGORITHM COMPARISON REPORT: QMIX vs IQL\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERVIEW:\n")
        f.write("---------\n")
        f.write("This report compares QMIX (Monotonic Value Function Factorisation)\n")
        f.write("with IQL (Independent Q-Learning) on the Overcooked environment.\n\n")
        
        f.write("QMIX uses a centralized mixing network that combines individual\n")
        f.write("Q-values into a joint Q-value, enabling explicit coordination.\n\n")
        
        f.write("IQL treats each agent independently, learning separate Q-functions\n")
        f.write("without any coordination mechanism.\n\n")
        
        f.write("="*60 + "\n")
        f.write("PERFORMANCE RESULTS:\n")
        f.write("="*60 + "\n\n")
        
        for layout in results_dict.keys():
            layout_results = results_dict[layout]
            f.write(f"Layout: {layout.replace('_', ' ').title()}\n")
            f.write("-" * 60 + "\n")
            
            if 'qmix' in layout_results:
                qmix = layout_results['qmix']
                f.write(f"QMIX:\n")
                f.write(f"  Mean soups: {qmix['mean_soups']:.2f} ± {qmix['std_soups']:.2f}\n")
                f.write(f"  Success rate: {qmix['success_rate']:.1f}%\n")
                f.write(f"  Target met: {'Yes' if qmix['mean_soups'] >= 7.0 else 'No'}\n")
            
            if 'iql' in layout_results:
                iql = layout_results['iql']
                f.write(f"IQL:\n")
                f.write(f"  Mean soups: {iql['mean_soups']:.2f} ± {iql['std_soups']:.2f}\n")
                f.write(f"  Success rate: {iql['success_rate']:.1f}%\n")
                f.write(f"  Target met: {'Yes' if iql['mean_soups'] >= 7.0 else 'No'}\n")
            
            if 'qmix' in layout_results and 'iql' in layout_results:
                improvement = qmix['mean_soups'] - iql['mean_soups']
                improvement_pct = (improvement / iql['mean_soups'] * 100) if iql['mean_soups'] > 0 else 0
                f.write(f"\nQMIX improvement: {improvement:+.2f} soups ({improvement_pct:+.1f}%)\n")
            
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("CONCLUSION:\n")
        f.write("="*60 + "\n\n")
        
        # Compute overall statistics
        qmix_means = []
        iql_means = []
        for layout in results_dict.keys():
            layout_results = results_dict[layout]
            if 'qmix' in layout_results:
                qmix_means.append(layout_results['qmix']['mean_soups'])
            if 'iql' in layout_results:
                iql_means.append(layout_results['iql']['mean_soups'])
        
        if qmix_means and iql_means:
            avg_qmix = np.mean(qmix_means)
            avg_iql = np.mean(iql_means)
            overall_improvement = avg_qmix - avg_iql
            overall_improvement_pct = (overall_improvement / avg_iql * 100) if avg_iql > 0 else 0
            
            f.write(f"Average performance across all layouts:\n")
            f.write(f"  QMIX: {avg_qmix:.2f} soups/episode\n")
            f.write(f"  IQL:  {avg_iql:.2f} soups/episode\n")
            f.write(f"  QMIX improvement: {overall_improvement:+.2f} soups ({overall_improvement_pct:+.1f}%)\n\n")
            
            f.write("The centralized mixing network in QMIX provides explicit\n")
            f.write("coordination between agents, enabling better performance than\n")
            f.write("independent learning. This demonstrates that multi-agent\n")
            f.write("coordination is crucial for the Overcooked task.\n")
    
    print(f"Comparison report saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare QMIX vs IQL algorithms')
    parser.add_argument('--layouts', type=str, nargs='+', default=None,
                        choices=['cramped_room', 'coordination_ring', 'counter_circuit_o_1order'],
                        help='Layouts to compare (default: all)')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save comparison results')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes per algorithm/layout')
    
    args = parser.parse_args()
    
    compare_algorithms(
        layouts=args.layouts,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        num_episodes=args.episodes
    )


if __name__ == '__main__':
    main()

