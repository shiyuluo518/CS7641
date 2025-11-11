"""
Evaluation script for trained QMIX agents.
Can evaluate single layout or all layouts and generate all required plots.
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
from src.algorithms.qmix import QMIX
from src.utils import plot_evaluation_results, plot_metrics, save_results
from src.generate_plots import generate_all_plots
import src.config as config


def evaluate_agent(layout_name, model_path, num_episodes=100):
    """
    Evaluate trained QMIX agents on a specific layout.
    
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
    
    # Get true global state dimension
    test_state = env.get_global_state()
    state_dim = len(test_state)
    
    # Initialize QMIX
    device = torch.device(config.DEVICE)
    qmix = QMIX(
        obs_dim=obs_dim,  # Use dynamic observation dimension (varies by layout)
        action_dim=config.ACTION_DIM,
        state_dim=state_dim,  # Use true global state dimension
        n_agents=config.N_AGENTS,
        hidden_dim=config.HIDDEN_DIM,
        device=device
    )
    
    # Load model
    qmix.load(model_path)
    for net in qmix.q_networks:
        net.eval()
    qmix.mixing_network.eval()
    
    # Evaluation metrics
    episode_soups = []
    episode_rewards = []
    all_metrics = []
    
    print(f"Evaluating on layout: {layout_name}")
    print(f"Running {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = [0.0, 0.0]
        steps = 0
        
        while not done and steps < config.HORIZON:
            # Select actions (no exploration during evaluation)
            actions = qmix.select_actions(obs, training=False)
            
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
    print(f"Evaluation Results for {layout_name}")
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
        print(f"\nRecommendations:")
        print(f"  - Increase training episodes (current: check training logs)")
        print(f"  - Tune hyperparameters (learning rate, epsilon decay)")
        print(f"  - Try custom reward shaping (set REWARD_SHAPING in config.py)")
        print(f"  - Check if training converged (review training curves)")
    print(f"{'='*60}")
    
    if aux_metrics:
        print(f"\nAuxiliary Metrics:")
        for key, stats in aux_metrics.items():
            print(f"  {key}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    results = {
        'layout': layout_name,
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


def evaluate_all_layouts(models_dir='models', results_dir='results', num_episodes=100, logs_dir='logs'):
    """Evaluate trained models on all three layouts."""
    layouts = ['cramped_room', 'coordination_ring', 'counter_circuit_o_1order']
    all_results = {}
    summary_results = {}
    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Evaluating on all layouts")
    print("="*60)
    
    for layout in layouts:
        model_path = Path(models_dir) / f"{layout}_final.pth"
        
        if not model_path.exists():
            print(f"\nWarning: Model not found for {layout}: {model_path}")
            print(f"Skipping {layout}...")
            summary_results[layout] = {'success': False, 'reason': 'Model not found'}
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {layout}")
        print(f"{'='*60}")
        
        results = evaluate_agent(
            layout_name=layout,
            model_path=str(model_path),
            num_episodes=num_episodes
        )
        
        all_results[layout] = results
        summary_results[layout] = {
            'mean_soups': results['mean_soups'],
            'success': results['success'],
            'success_rate': results['success_rate']
        }
        save_results(results, Path(results_dir) / f"{layout}_evaluation.json")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    all_success = True
    for layout in layouts:
        if layout in summary_results:
            result = summary_results[layout]
            if 'reason' in result:
                print(f"{layout:30s}: {result['reason']}")
                all_success = False
            else:
                status = "✓ PASS" if result['success'] else "✗ FAIL"
                print(f"{layout:30s}: {status} | Mean: {result['mean_soups']:.2f} | Success Rate: {result['success_rate']:.1f}%")
                if not result['success']:
                    all_success = False
        else:
            print(f"{layout:30s}: NOT EVALUATED")
            all_success = False
    
    print(f"{'='*60}")
    if all_success:
        print("✓ ALL LAYOUTS MEET TARGET (≥7 soups/episode)")
    else:
        print("✗ SOME LAYOUTS FAILED TO MEET TARGET")
        print("\nRecommendations:")
        print("  - Review training curves to check convergence")
        print("  - Consider increasing training episodes for failed layouts")
        print("  - Try tuning hyperparameters or reward shaping")
    print(f"{'='*60}\n")
    
    # Save summary
    summary_path = Path(results_dir) / "evaluation_summary.json"
    save_results({
        'timestamp': str(Path(__file__).stat().st_mtime),
        'target_soups': 7.0,
        'num_episodes': num_episodes,
        'results': summary_results,
        'all_success': all_success
    }, summary_path)
    print(f"Summary saved to: {summary_path}")
    
    # Generate all required plots (saves evaluation results first)
    generate_all_plots(results_dir, logs_dir=logs_dir)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained QMIX agents')
    parser.add_argument('--layout', type=str, default=None,
                        choices=['cramped_room', 'coordination_ring', 'counter_circuit_o_1order', None],
                        help='Layout to evaluate on (omit to evaluate all)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model checkpoint (required if --layout specified)')
    parser.add_argument('--episodes', type=int, default=config.EVAL_EPISODES,
                        help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing trained models (for --all)')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory containing training logs')
    parser.add_argument('--all', action='store_true',
                        help='Evaluate all layouts and generate all plots')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if args.all or args.layout is None:
        # Evaluate all layouts
        try:
            all_results = evaluate_all_layouts(
                models_dir=args.models_dir,
                results_dir=args.save_dir,
                num_episodes=args.episodes,
                logs_dir=args.logs_dir
            )
            # Generate plots even if evaluation had issues
            generate_all_plots(args.save_dir, logs_dir=args.logs_dir)
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {e}")
            print("Attempting to generate plots from existing results...")
            import traceback
            traceback.print_exc()
            # Still try to generate plots from existing data
            generate_all_plots(args.save_dir, args.logs_dir)
    else:
        # Evaluate single layout
        if args.model_path is None:
            args.model_path = Path(args.models_dir) / f"{args.layout}_final.pth"
        
        results = evaluate_agent(
            layout_name=args.layout,
            model_path=str(args.model_path),
            num_episodes=args.episodes
        )
        
        # Save results
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        save_results(results, Path(args.save_dir) / f"{args.layout}_evaluation.json")
        
        # Plot results
        plot_evaluation_results(
            {args.layout: results['episode_soups']},
            save_path=Path(args.save_dir) / f"{args.layout}_evaluation_curve.png"
        )
        
        # Plot auxiliary metrics for evaluation
        if results.get('auxiliary_metrics_episode'):
            plot_metrics(
                results['auxiliary_metrics_episode'],
                save_path=Path(args.save_dir) / f"{args.layout}_eval_auxiliary_metrics.png",
                title=f"Evaluation Auxiliary Metrics - {args.layout}"
            )


if __name__ == '__main__':
    main()

