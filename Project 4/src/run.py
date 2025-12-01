import yaml
import time
import torch
import datetime
import numpy as np
from loguru import logger
from munch import munchify
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.agents import RandomAgent, MyFancyAgent
from src.utils import (
    device,
    set_seed,
    make_environment,
    evaluate,
    demo,
    get_world_name,
    get_race_type,
    ENVIRONMENT_PARAMS_PATH,
)


DEVICE = device()
HYPER_PARAMS_PATH: str='configs/hyper_params.yaml'

# Track names for multi-track training
TRACK_NAMES = [
    'reInvent2019_wide',    # A to Z Speedway
    'reInvent2019_track',   # Smile Speedway
    'Vegas_track',          # AWS Summit Raceway
]


def tensor(x: np.array, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.tensor(x, dtype=type, device=device)


def zeros(x: tuple, type=torch.float, device=DEVICE) -> torch.Tensor:
    return torch.zeros(x, dtype=type, device=device)


def update_environment_config(world_name: str):
    """
    Update the environment config file with the specified world name.
    This allows training on different tracks.
    """
    try:
        with open(ENVIRONMENT_PARAMS_PATH, 'r') as f:
            env_params = yaml.safe_load(f)
        
        env_params['WORLD_NAME'] = world_name
        
        with open(ENVIRONMENT_PARAMS_PATH, 'w') as f:
            yaml.dump(env_params, f, default_flow_style=False)
        
        logger.info(f"Updated environment config to use track: {world_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to update environment config: {e}")
        return False


def run(hparams, multi_track=False, tracks=None):
    """
    Enhanced training function with improved error handling and evaluation.
    
    Args:
        hparams: Dictionary of hyperparameters
        multi_track: If True, train on multiple tracks sequentially
        tracks: List of track names to train on (default: all 3 tracks)
    """
    start_time = time.time()
    
    # load hyper-params if not provided
    with open(HYPER_PARAMS_PATH, 'r') as file:
        default_hparams = yaml.safe_load(file)
    
    final_hparams = default_hparams.copy()
    final_hparams.update(hparams)
    args = munchify(final_hparams)
    
    # Determine tracks to train on
    if tracks is None:
        tracks = TRACK_NAMES if multi_track else [get_world_name()]
    
    logger.info(f"Training on tracks: {tracks}")
    
    # save parameters and/or configs if you wish
    run_name = (
        f"{args.environment}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    )
    writer = SummaryWriter(f"results/runs/{run_name}")
    writer.add_text(
        'hyperparameters',
        "|param|value|\n|-|-|\n%s" % (
            "\n".join(
                [f"|{key}|{value}|" for key, value in vars(args).items()]
            )
        ),
    )
    writer.add_text('training_config', f"Multi-track: {multi_track}\nTracks: {tracks}")
    
    set_seed(args.seed)
    
    # Convert string values from YAML to appropriate types
    learning_rate = float(getattr(args, 'learning_rate', 3e-4))
    n_steps = int(getattr(args, 'n_steps', 2048))
    batch_size = int(getattr(args, 'batch_size', 64))
    n_epochs = int(getattr(args, 'n_epochs', 10))
    gamma = float(getattr(args, 'gamma', 0.99))
    
    # Create agent (will be reused across tracks if multi_track)
    agent = None
    total_timesteps_per_track = int(args.total_timesteps) // len(tracks) if multi_track else int(args.total_timesteps)
    
    # Train on each track
    for track_idx, track_name in enumerate(tracks):
        logger.info(f"\n{'='*70}")
        logger.info(f"Training on track {track_idx + 1}/{len(tracks)}: {track_name}")
        logger.info(f"{'='*70}")
        
        # Update environment config for this track
        if multi_track:
            update_environment_config(track_name)
            # Need to recreate environment after config change
            # Note: In practice, you may need to restart the simulation container
            logger.warning("For multi-track training, ensure simulation container is restarted with new config")
        
        env = make_environment(args.environment)
        
        # Create agent on first track, reuse on subsequent tracks
        if agent is None:
            agent = MyFancyAgent(
                environment=env,
                name=args.experiment_name,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                verbose=1
            )
        else:
            # For multi-track training, continue training the same agent
            # The environment wrapper handles track changes
            logger.info(f"Continuing training on {track_name} with existing agent...")
        
        # Train the agent on this track
        logger.info(f'Starting training for {total_timesteps_per_track} timesteps on {track_name}...')
        
        save_path = f'models/{args.experiment_name}_{track_name}_{int(time.time())}.zip'
        agent.train(
            total_timesteps=total_timesteps_per_track,
            log_interval=10,
            save_path=save_path,
            verbose=1
        )
        
        logger.info(f'Training on {track_name} completed. Model saved to {save_path}')
        
        # Log track-specific metrics
        writer.add_scalar(f'track/{track_name}/training_completed', 1, track_idx)
        
        env.close()
    
    # Final model save
    final_save_path = f'models/{args.experiment_name}_final_{int(time.time())}.zip'
    agent.save(final_save_path)
    logger.info(f'Final model saved to {final_save_path}')
    
    # Evaluate the trained agent using src.utils.evaluate
    logger.info('\n' + '='*70)
    logger.info('Starting comprehensive evaluation on all tracks...')
    logger.info('='*70)
    
    eval_metrics = None
    try:
        # Use the provided evaluate function
        eval_metrics = evaluate(agent, environment_name=args.environment)
        
        logger.info('\nEvaluation Results:')
        for track_name, metrics in eval_metrics.items():
            logger.info(f'\n{track_name}:')
            logger.info(f"  Progress: {np.mean(metrics['progress']):.1f}% ± {np.std(metrics['progress']):.1f}%")
            logger.info(f"  Lap Time: {np.nanmean(metrics['lap_time']):.2f}s (mean, excluding NaN)")
            
            # Log to TensorBoard
            writer.add_scalar(f'eval/{track_name}/mean_progress', np.mean(metrics['progress']), args.total_timesteps)
            writer.add_scalar(f'eval/{track_name}/mean_lap_time', np.nanmean(metrics['lap_time']), args.total_timesteps)
        
        # Check if problem is solved (100% progress for 5 consecutive episodes on all tracks)
        solved = True
        for track_name, metrics in eval_metrics.items():
            progress_values = metrics['progress']
            # Check last 5 episodes
            if len(progress_values) >= 5:
                last_5 = progress_values[-5:]
                if not all(p >= 100.0 for p in last_5):
                    solved = False
                    logger.warning(f"{track_name}: Not solved (last 5 episodes: {last_5})")
            else:
                solved = False
        
        if solved:
            logger.info('\n[SUCCESS] Problem solved! Agent completes 100% progress on all tracks!')
        else:
            logger.info('\n[INFO] Problem not yet solved. Continue training to improve performance.')
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        logger.info("Falling back to simple evaluation...")
        import traceback
        traceback.print_exc()
        
        try:
            # Try simple evaluation with better error handling
            eval_results = agent.evaluate(n_episodes=5, deterministic=True)
            logger.info(
                f'\nSimple evaluation results:\n'
                f'  Mean reward: {eval_results["mean_reward"]:.2f} ± {eval_results["std_reward"]:.2f}\n'
                f'  Mean episode length: {eval_results["mean_length"]:.2f} ± {eval_results["std_length"]:.2f}\n'
                f'  Mean progress: {eval_results.get("mean_progress", 0):.1f}% ± {eval_results.get("std_progress", 0):.1f}%'
            )
            
            # Log to TensorBoard if available
            try:
                writer.add_scalar('eval/simple/mean_reward', eval_results['mean_reward'], args.total_timesteps)
                writer.add_scalar('eval/simple/mean_progress', eval_results.get('mean_progress', 0), args.total_timesteps)
            except:
                pass
                
        except Exception as e2:
            logger.error(f"Simple evaluation also failed: {e2}")
            logger.warning("Skipping evaluation. You can evaluate manually later.")
            import traceback
            traceback.print_exc()
    
    # Generate demo video using src.utils.demo
    logger.info('\n' + '='*70)
    logger.info('Generating demo video...')
    logger.info('='*70)
    
    try:
        demo(agent, environment_name=args.environment, directory='./results/demos')
        logger.info('Demo video saved to ./results/demos/')
    except Exception as e:
        logger.warning(f"Demo generation failed: {e}")
        logger.info("You can generate demo videos manually using: demo(agent)")
        import traceback
        traceback.print_exc()
    
    # Log final summary with performance analysis
    training_time = time.time() - start_time
    writer.add_scalar('training/total_time_seconds', training_time, args.total_timesteps)
    
    logger.info(f'\nTotal training time: {training_time/60:.1f} minutes')
    
    # Performance summary
    logger.info('\n' + '='*70)
    logger.info('Training Performance Summary')
    logger.info('='*70)
    logger.info(f'Total Timesteps: {args.total_timesteps:,}')
    logger.info(f'Training Time: {training_time/60:.1f} minutes')
    logger.info(f'Final Model: {final_save_path}')
    
    if eval_metrics:
        logger.info('\nEvaluation Summary:')
        for track_name, metrics in eval_metrics.items():
            mean_progress = np.mean(metrics['progress'])
            logger.info(f'  {track_name}: {mean_progress:.1f}% progress')
    
    writer.close()
    
    logger.info(f'\nTraining complete! Model saved to: {final_save_path}')
    logger.info(f'TensorBoard logs: results/runs/{run_name}')
    logger.info(f'Evaluation results: results/evaluations/')
    logger.info(f'Demo videos: results/demos/')
