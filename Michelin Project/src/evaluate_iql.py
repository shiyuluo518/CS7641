"""
Evaluation script for trained IQL agents.
"""

import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch

from src.env_wrapper import OvercookedWrapper
from src.algorithms.iql import IQL
from src.utils import plot_evaluation_results, save_results
import src.config as config


def evaluate_agent(layout_name, model_path, num_episodes=100):
    """
    Evaluate trained IQL agents on a specific layout.
    
    Args:
        layout_name: Name of the layout to evaluate on
        model_path: Path to saved model checkpoint
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation results
    """
    # Initialize environment
    env = OvercookedWrapper(layout_name)
    
    # Get observation dimension dynamically from environment (varies by layout)
    test_obs = env.reset()
    obs_dim = len(test_obs[0]) if isinstance(test_obs, list) else len(test_obs)
    
    # Initialize IQL
    device = torch.device(config.DEVICE)
    iql = IQL(
        obs_dim=obs_dim,  # Use dynamic observation dimension (varies by layout)
        action_dim=config.ACTION_DIM,
        n_agents=config.N_AGENTS,
        hidden_dim=config.HIDDEN_DIM,
        device=device
    )
    
    # Load model
    iql.load(model_path)
    for net in iql.q_networks:
        net.eval()
    
    # Evaluation metrics
    episode_soups = []
    episode_rewards = []
    all_metrics = []
    
    print(f"Evaluating IQL on layout: {layout_name}")
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = [0.0, 0.0]
        steps = 0
        
        while not done and steps < config.HORIZON:
            # Select actions (no exploration during evaluation)
            actions = iql.select_actions(obs, training=False)
            
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
            all_metrics.append(ep_info)
        elif done:
            # Episode ended but no episode info - append 0 soups
            episode_soups.append(0)
        
        episode_rewards.append(sum(episode_reward))
        
        if (episode + 1) % 10 == 0:
            recent_mean = np.mean(episode_soups[-10:]) if episode_soups else 0
            print(f"Episode {episode+1}: Recent avg soups = {recent_mean:.2f}")
    
    # Compute statistics
    mean_soups = np.mean(episode_soups)
    std_soups = np.std(episode_soups)
    min_soups = np.min(episode_soups)
    max_soups = np.max(episode_soups)
    
    # Compute additional metrics if available
    aux_metrics = {}
    aux_metrics_episode = {}
    if all_metrics:
        # Get all metric keys
        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        
        for key in all_keys:
            if key != 'soup_delivered':
                values = [m.get(key, 0) for m in all_metrics]
                aux_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
                aux_metrics_episode[key] = values
    
    # Validation: Check if performance meets target
    target_soups = 7.0
    success = mean_soups >= target_soups
    success_rate = (np.array(episode_soups) >= target_soups).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {layout_name} (IQL)")
    print(f"{'='*60}")
    print(f"Mean soups delivered: {mean_soups:.2f} ± {std_soups:.2f}")
    print(f"Min: {min_soups}, Max: {max_soups}")
    print(f"Target: ≥{target_soups} soups per episode")
    print(f"Episodes meeting target: {success_rate:.1f}%")
    print(f"\n{'='*60}")
    if success:
        print(f"✓ SUCCESS: Mean soups ({mean_soups:.2f}) meets target (≥{target_soups})")
    else:
        print(f"✗ FAILURE: Mean soups ({mean_soups:.2f}) below target (≥{target_soups})")
    print(f"{'='*60}")
    
    results = {
        'layout': layout_name,
        'algorithm': 'iql',
        'num_episodes': num_episodes,
        'mean_soups': float(mean_soups),
        'std_soups': float(std_soups),
        'min_soups': int(min_soups),
        'max_soups': int(max_soups),
        'target_soups': target_soups,
        'success': success,
        'success_rate': float(success_rate),
        'episode_soups': episode_soups,
        'episode_rewards': episode_rewards,
        'auxiliary_metrics': aux_metrics,
        'auxiliary_metrics_episode': aux_metrics_episode,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained IQL agents')
    parser.add_argument('--layout', type=str, required=True,
                        choices=['cramped_room', 'coordination_ring', 'counter_circuit_o_1order'],
                        help='Layout to evaluate on')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--episodes', type=int, default=config.EVAL_EPISODES,
                        help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    results = evaluate_agent(
        layout_name=args.layout,
        model_path=args.model_path,
        num_episodes=args.episodes
    )
    
    # Save results
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_results(results, Path(args.save_dir) / f"{args.layout}_iql_evaluation.json")
    
    # Plot results
    plot_evaluation_results(
        {args.layout: results['episode_soups']},
        save_path=Path(args.save_dir) / f"{args.layout}_iql_evaluation_curve.png"
    )


if __name__ == '__main__':
    main()

