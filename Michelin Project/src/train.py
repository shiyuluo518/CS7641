"""
Training script for multi-agent Overcooked environment using QMIX.

QUICK MODE: This script uses QUICK_MODE from src/config.py (default: True).
- QUICK_MODE = True: Faster training (~1.5-2.5 hours for all layouts)
- QUICK_MODE = False: Full training for ≥7.0 performance guarantee (~10-13 hours)
Episode counts are automatically set based on QUICK_MODE in config.py.
"""

import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.env_wrapper import OvercookedWrapper
from src.algorithms.qmix import QMIX
from src.algorithms.replay_buffer import MultiAgentReplayBuffer
from src.utils import MetricsTracker, plot_training_curves, save_results
import src.config as config

def train_agent(layout_name, episodes=15000, save_dir='models', log_dir='logs'):
    """
    Train QMIX agents for a specific layout.
    
    Args:
        layout_name: Name of the layout to train on
        episodes: Number of training episodes
        save_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
    """
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup reward shaping
    reward_shaping = None
    if config.REWARD_SHAPING == 'collaborative':
        reward_shaping = config.get_collaborative_reward_shaping()
        print(f"Using collaborative reward shaping: {reward_shaping}")
    elif config.REWARD_SHAPING == 'efficient':
        reward_shaping = config.get_efficiency_reward_shaping()
        print(f"Using efficiency reward shaping: {reward_shaping}")
    
    # Initialize environment (horizon from config, optimized for quick mode)
    env = OvercookedWrapper(layout_name, reward_shaping=reward_shaping, horizon=config.HORIZON)
    
    # Get observation dimension dynamically from environment (varies by layout)
    test_obs = env.reset()
    obs_dim = len(test_obs[0]) if isinstance(test_obs, list) else len(test_obs)
    
    # Get true global state dimension from environment
    test_state = env.get_global_state()
    state_dim = len(test_state)
    
    # Initialize QMIX
    device = torch.device(config.DEVICE)
    
    # Print device information for GPU acceleration
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("GPU acceleration enabled - expect 3-10x speedup!")
    else:
        print("Using CPU - GPU not available. Install CUDA-enabled PyTorch for faster training.")
    
    qmix = QMIX(
        obs_dim=obs_dim,  # Use dynamic observation dimension (varies by layout)
        action_dim=config.ACTION_DIM,
        state_dim=state_dim,  # Use true global state dimension
        n_agents=config.N_AGENTS,
        hidden_dim=config.HIDDEN_DIM,
        lr=config.LEARNING_RATE,
        gamma=config.GAMMA,
        epsilon_start=config.EPSILON_START,
        epsilon_end=config.EPSILON_END,
        epsilon_decay=config.EPSILON_DECAY,
        target_update_interval=config.TARGET_UPDATE_INTERVAL,
        device=device
    )
    
    print(f"Observation dimension: {obs_dim} (per agent)")
    print(f"Global state dimension: {state_dim} (vs {obs_dim * 2} from concatenation)")
    
    # Initialize replay buffer with fixed state dimension (on same device as model)
    replay_buffer = MultiAgentReplayBuffer(capacity=config.BUFFER_SIZE, state_dim=state_dim, device=device)
    
    # Metrics tracking
    metrics = MetricsTracker()
    episode_rewards = []
    episode_soups = []
    training_losses = []
    # Auxiliary metrics for analysis (skip in quick mode for speed)
    track_auxiliary = not config.QUICK_MODE  # Skip auxiliary metrics in quick mode
    auxiliary_metrics = {
        'onion_pickups': [],
        'dish_pickups': [],
        'soup_pickups': [],
        'placements_in_pot': [],
        'dropped_items': [],
    } if track_auxiliary else {}
    
    # Training loop
    print(f"Training on layout: {layout_name}")
    print(f"Total episodes: {episodes}")
    print(f"Device: {device}")
    print(f"Hyperparameters: lr={config.LEARNING_RATE}, gamma={config.GAMMA}, batch_size={config.BATCH_SIZE}")
    
    for episode in tqdm(range(episodes)):
        obs = env.reset()
        done = False
        episode_reward = [0.0, 0.0]
        steps = 0
        
        # Get initial global state
        state = env.get_global_state()
        
        while not done and steps < config.HORIZON:
            # Select actions (GPU-accelerated, fast)
            actions = qmix.select_actions(obs, training=True)
            
            # Step environment (CPU-bound bottleneck - this is the slow part)
            next_obs, rewards, done, info = env.step(actions)
            
            # Get next global state (CPU-bound, needed for QMIX mixing network)
            next_state = env.get_global_state()
            
            # Store transition in replay buffer
            replay_buffer.push(obs, actions, rewards, next_obs, done, state, next_state)
            
            # Update episode rewards
            episode_reward[0] += rewards[0]
            episode_reward[1] += rewards[1]
            
            # Train if we have enough samples (GPU-accelerated, fast)
            if len(replay_buffer) >= config.BATCH_SIZE and steps % config.TRAIN_EVERY_N_STEPS == 0:
                batch = replay_buffer.sample(config.BATCH_SIZE)
                # Batch is already on device (from replay buffer), no need to move again
                loss = qmix.update(batch, batch['states'], batch['next_states'])
                training_losses.append(loss)
            
            obs = next_obs
            state = next_state
            steps += 1
        
        # Track metrics (only when episode is done)
        if done and isinstance(info, dict) and 'episode' in info:
            ep_info = info['episode']
            ep_game_stats = ep_info.get('ep_game_stats', {})
            
            # Extract soup deliveries from ep_game_stats
            # soup_delivery is a list of lists: [[agent0_timesteps], [agent1_timesteps]]
            soup_delivery = ep_game_stats.get('soup_delivery', [[], []])
            # Count total number of soup deliveries across all agents
            soups = sum(len(deliveries) for deliveries in soup_delivery) if isinstance(soup_delivery, (list, tuple)) else 0
            episode_soups.append(soups)
            
            # Only update metrics tracker if not in quick mode (saves time)
            if not config.QUICK_MODE:
                metrics.update(episode, info)
            
            # Track auxiliary metrics only if enabled (skip in quick mode for speed)
            if track_auxiliary:
                # Track auxiliary metrics from ep_game_stats
                # Most metrics are lists of lists: [[agent0_timesteps], [agent1_timesteps]]
                onion_pickup = ep_game_stats.get('onion_pickup', [[], []])
                if isinstance(onion_pickup, (list, tuple)):
                    onion_pickup = sum(len(pickups) for pickups in onion_pickup)
                else:
                    onion_pickup = int(onion_pickup) if onion_pickup else 0
                auxiliary_metrics['onion_pickups'].append(onion_pickup)
                
                dish_pickup = ep_game_stats.get('dish_pickup', [[], []])
                if isinstance(dish_pickup, (list, tuple)):
                    dish_pickup = sum(len(pickups) for pickups in dish_pickup)
                else:
                    dish_pickup = int(dish_pickup) if dish_pickup else 0
                auxiliary_metrics['dish_pickups'].append(dish_pickup)
                
                soup_pickup = ep_game_stats.get('soup_pickup', [[], []])
                if isinstance(soup_pickup, (list, tuple)):
                    soup_pickup = sum(len(pickups) for pickups in soup_pickup)
                else:
                    soup_pickup = int(soup_pickup) if soup_pickup else 0
                auxiliary_metrics['soup_pickups'].append(soup_pickup)
                
                # Count potting actions (onion + tomato)
                potting_onion = ep_game_stats.get('potting_onion', [[], []])
                potting_tomato = ep_game_stats.get('potting_tomato', [[], []])
                if isinstance(potting_onion, (list, tuple)):
                    placement = sum(len(pottings) for pottings in potting_onion)
                else:
                    placement = int(potting_onion) if potting_onion else 0
                if isinstance(potting_tomato, (list, tuple)):
                    placement += sum(len(pottings) for pottings in potting_tomato)
                else:
                    placement += int(potting_tomato) if potting_tomato else 0
                auxiliary_metrics['placements_in_pot'].append(placement)
                
                # Count dropped items (soup_drop)
                soup_drop = ep_game_stats.get('soup_drop', [[], []])
                if isinstance(soup_drop, (list, tuple)):
                    dropped = sum(len(drops) for drops in soup_drop)
                else:
                    dropped = int(soup_drop) if soup_drop else 0
                auxiliary_metrics['dropped_items'].append(dropped)
        elif done:
            # Episode ended but no episode info - append 0 soups
            episode_soups.append(0)
        
        episode_rewards.append(sum(episode_reward))
        
        # Progress monitoring: Save intermediate results every 200 episodes
        # This allows progress checking without interrupting training
        if (episode + 1) % 200 == 0:
            intermediate_results = {
                'layout': layout_name,
                'episodes': episode + 1,
                'soups_delivered': episode_soups,
                'episode_rewards': episode_rewards,
                'training_losses': training_losses,
                'final_epsilon': qmix.epsilon,
            }
            save_results(intermediate_results, Path(log_dir) / f"{layout_name}_training_results.json")
            print(f"  [PROGRESS] Saved progress at episode {episode+1} - run 'python check_training_progress.py' to monitor")
        
        # Logging and convergence monitoring (every 100 episodes for progress tracking)
        log_interval = 100  # Log every 100 episodes for progress monitoring
        if (episode + 1) % log_interval == 0:
            recent_soups = np.mean(episode_soups[-100:]) if episode_soups else 0
            recent_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            recent_loss = np.mean(training_losses[-100:]) if training_losses else 0
            
            # Progress percentage
            progress_pct = ((episode + 1) / episodes) * 100
            
            # Convergence check: compare last 100 vs previous 100
            convergence_info = ""
            if len(episode_soups) >= 200:
                prev_100_soups = np.mean(episode_soups[-200:-100])
                improvement = recent_soups - prev_100_soups
                if abs(improvement) < 0.1:
                    convergence_info = " (CONVERGED)" if recent_soups >= 7.0 else " (PLATEAUED)"
                elif improvement < -0.5:
                    convergence_info = " (DECREASING)"
                elif improvement > 0.5:
                    convergence_info = " (IMPROVING)"
            
            target_met = "[PASS]" if recent_soups >= 7.0 else "[...]"
            print(f"\nEpisode {episode+1}/{episodes} ({progress_pct:.1f}%):")
            print(f"  Recent avg soups: {recent_soups:.2f} {target_met} (target: >=7.0){convergence_info}")
            print(f"  Recent avg reward: {recent_reward:.2f}")
            print(f"  Recent avg loss: {recent_loss:.4f}")
            print(f"  Epsilon: {qmix.epsilon:.3f}")
            print(f"  Buffer size: {len(replay_buffer)}/{config.BUFFER_SIZE}")
        
        # Save model checkpoint for recovery (every 500 episodes in quick mode)
        checkpoint_interval = 500 if config.QUICK_MODE else 1000
        if (episode + 1) % checkpoint_interval == 0:
            qmix.save(Path(save_dir) / f"{layout_name}_checkpoint_{episode+1}.pth")
            print(f"  [CHECKPOINT] Saved checkpoint at episode {episode+1}")
    
    # Save final model
    qmix.save(Path(save_dir) / f"{layout_name}_final.pth")
    
    # Save training results
    results = {
        'layout': layout_name,
        'episodes': episodes,
        'soups_delivered': episode_soups,
        'episode_rewards': episode_rewards,
        'training_losses': training_losses,
        'final_epsilon': qmix.epsilon,
        'auxiliary_metrics': auxiliary_metrics,
    }
    save_results(results, Path(log_dir) / f"{layout_name}_training_results.json")
    
    # Plot training curves (always generate plots)
    try:
        plot_training_curves(
            {layout_name: episode_soups},
            save_path=Path(log_dir) / f"{layout_name}_training_curve.png"
        )
        print(f"   [OK] Training curve saved: {log_dir}/{layout_name}_training_curve.png")
    except Exception as e:
        print(f"   [WARNING] Could not generate training curve: {e}")
        import traceback
        traceback.print_exc()
    
    # Plot auxiliary metrics (skip in quick mode)
    if track_auxiliary and auxiliary_metrics:
        try:
            from src.utils import plot_metrics
            plot_metrics(
                auxiliary_metrics,
                save_path=Path(log_dir) / f"{layout_name}_auxiliary_metrics.png",
                title=f"Auxiliary Metrics - {layout_name}"
            )
            print(f"   [OK] Auxiliary metrics plot saved: {log_dir}/{layout_name}_auxiliary_metrics.png")
        except Exception as e:
            print(f"   [WARNING] Could not generate auxiliary metrics plot: {e}")
            import traceback
            traceback.print_exc()
    elif config.QUICK_MODE:
        print(f"   [SKIP] Auxiliary metrics skipped in quick mode for speed")
    
    # Final performance analysis
    final_100_soups = np.mean(episode_soups[-100:]) if len(episode_soups) >= 100 else np.mean(episode_soups)
    final_1000_soups = np.mean(episode_soups[-1000:]) if len(episode_soups) >= 1000 else final_100_soups
    target_met = final_100_soups >= 7.0
    
    print(f"\n{'='*60}")
    print(f"Training completed for {layout_name}!")
    print(f"{'='*60}")
    print(f"Total episodes: {episodes}")
    print(f"Final performance (last 100 episodes): {final_100_soups:.2f} {'[PASS]' if target_met else '[FAIL]'}")
    print(f"Final performance (last 1000 episodes): {final_1000_soups:.2f}")
    if len(episode_soups) >= 100:
        best_window = max([np.mean(episode_soups[i:i+100]) for i in range(len(episode_soups)-99)])
        print(f"Best 100-episode window: {best_window:.2f}")
    else:
        print(f"Best episode: {np.max(episode_soups):.2f}")
    print(f"Target: >=7.0 soups per episode")
    
    if target_met:
        print(f"\n[SUCCESS] Training successful: Mean soups ({final_100_soups:.2f}) meets target!")
    else:
        print(f"\n[WARNING] Training incomplete: Mean soups ({final_100_soups:.2f}) below target.")
        print(f"Recommendations:")
        print(f"  - Train for more episodes (current: {episodes})")
        print(f"  - Check if learning rate needs adjustment")
        print(f"  - Consider using reward shaping (set REWARD_SHAPING in config.py)")
        print(f"  - Verify environment setup and reward signals")
    print(f"{'='*60}\n")
    
    # Add training success flag to results
    results['training_success'] = target_met
    results['final_100_soups'] = float(final_100_soups)
    results['final_1000_soups'] = float(final_1000_soups)
    if len(episode_soups) >= 100:
        results['best_window_100'] = float(max([np.mean(episode_soups[i:i+100]) for i in range(len(episode_soups)-99)]))
    else:
        results['best_episode'] = float(np.max(episode_soups))
    
    return qmix, results


def main():
    parser = argparse.ArgumentParser(
        description='Train multi-agent Overcooked agents with QMIX',
        epilog=f'QUICK_MODE: {config.QUICK_MODE} (edit src/config.py to change). '
               f'Default episodes: cramped_room={config.EPISODES_CRAMPED_ROOM}, '
               f'coordination_ring={config.EPISODES_COORDINATION_RING}, '
               f'counter_circuit_o_1order={config.EPISODES_COUNTER_CIRCUIT}')
    parser.add_argument('--layout', type=str, required=True,
                        choices=['cramped_room', 'coordination_ring', 'counter_circuit_o_1order'],
                        help='Layout to train on')
    parser.add_argument('--episodes', type=int, default=None,
                        help=f'Number of training episodes (default: layout-specific from config, QUICK_MODE={config.QUICK_MODE})')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Set default episodes based on layout (uses QUICK_MODE from config)
    if args.episodes is None:
        if args.layout == 'cramped_room':
            args.episodes = config.EPISODES_CRAMPED_ROOM
        elif args.layout == 'coordination_ring':
            args.episodes = config.EPISODES_COORDINATION_RING
        elif args.layout == 'counter_circuit_o_1order':
            args.episodes = config.EPISODES_COUNTER_CIRCUIT
        print(f"Using default episodes from config (QUICK_MODE={config.QUICK_MODE}): {args.episodes}")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train agent
    train_agent(
        layout_name=args.layout,
        episodes=args.episodes,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()

