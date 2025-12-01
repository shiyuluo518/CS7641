"""
Evaluate Time-Trial agent as baseline on Object-Avoidance and Head-to-Bot tasks.

This script evaluates the trained Time-Trial agent on:
1. Object-Avoidance task (6 obstacles)
2. Head-to-Bot task (3 bot cars)

This provides a baseline performance for comparison with Part II trained agents.
"""
import os
import sys
import yaml
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils import make_environment, evaluate_track, get_world_name, ENVIRONMENT_PARAMS_PATH
from src.agents import MyFancyAgent

def evaluate_time_trial_baseline(model_path, race_type, track_name='reInvent2019_wide', obstacles=0, bot_cars=0):
    """
    Evaluate Time-Trial agent on a different race type as baseline.
    
    Args:
        model_path: Path to Time-Trial model
        race_type: 'obstacle_avoidance' or 'head_to_bot'
        track_name: Track to evaluate on
        obstacles: Number of obstacles (6 for Object-Avoidance)
        bot_cars: Number of bot cars (3 for Head-to-Bot)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Baseline Evaluation: Time-Trial Agent on {race_type.replace('_', ' ').title()}")
    logger.info(f"{'='*70}")
    
    # Update environment configuration
    try:
        with open(ENVIRONMENT_PARAMS_PATH, 'r') as f:
            env_params = yaml.safe_load(f)
        
        env_params['WORLD_NAME'] = track_name
        env_params['NUMBER_OF_OBSTACLES'] = obstacles
        env_params['NUMBER_OF_BOT_CARS'] = bot_cars
        
        with open(ENVIRONMENT_PARAMS_PATH, 'w') as f:
            yaml.dump(env_params, f, default_flow_style=False)
        
        logger.info(f"Updated config: {track_name}, {obstacles} obstacles, {bot_cars} bot cars")
    except Exception as e:
        logger.error(f"Failed to update environment config: {e}")
        raise
    
    # Create environment and agent
    env = make_environment('deepracer-v0')
    agent = MyFancyAgent(environment=env, name='time_trial_baseline')
    
    # Load Time-Trial model
    agent.load(model_path)
    logger.info(f"Loaded Time-Trial model: {model_path}")
    
    # Evaluate on the track
    try:
        logger.info(f"Evaluating Time-Trial agent on {race_type} task...")
        logger.warning("Note: This agent was trained for Time-Trial, not {race_type}")
        logger.warning("Performance may be poor as it hasn't learned obstacle/bot avoidance")
        
        eval_metrics = evaluate_track(
            agent=agent,
            world_name=track_name,
            environment_name='deepracer-v0',
            directory='./results/evaluations'
        )
        
        logger.info(f"\nBaseline Evaluation Results ({race_type}):")
        logger.info(f"  Progress: {eval_metrics['progress']}")
        logger.info(f"  Lap Time: {eval_metrics['lap_time']}")
        
        return eval_metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        env.close()

def main():
    """Evaluate Time-Trial agent as baseline on both Part II tasks."""
    logger.info("="*70)
    logger.info("Time-Trial Baseline Evaluation on Part II Tasks")
    logger.info("="*70)
    
    # Find latest Time-Trial model
    models_dir = Path("models")
    time_trial_models = list(models_dir.glob("*time_trial*.zip"))
    
    if not time_trial_models:
        logger.error("No Time-Trial model found!")
        logger.error("Please train a Time-Trial model first using:")
        logger.error("  python scripts/training/train_iteration1_baseline.py")
        return
    
    # Get latest Time-Trial model
    latest_model = max(time_trial_models, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using Time-Trial model: {latest_model.name}")
    
    track_name = 'reInvent2019_wide'
    
    # Evaluate on Object-Avoidance task
    logger.info("\n" + "="*70)
    logger.info("1. Evaluating Time-Trial Agent on Object-Avoidance Task")
    logger.info("="*70)
    obstacle_metrics = evaluate_time_trial_baseline(
        model_path=str(latest_model),
        race_type='obstacle_avoidance',
        track_name=track_name,
        obstacles=6,
        bot_cars=0
    )
    
    # Evaluate on Head-to-Bot task
    logger.info("\n" + "="*70)
    logger.info("2. Evaluating Time-Trial Agent on Head-to-Bot Task")
    logger.info("="*70)
    head_to_bot_metrics = evaluate_time_trial_baseline(
        model_path=str(latest_model),
        race_type='head_to_bot',
        track_name=track_name,
        obstacles=0,
        bot_cars=3
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Baseline Evaluation Summary")
    logger.info("="*70)
    
    if obstacle_metrics:
        mean_progress = sum(obstacle_metrics['progress']) / len(obstacle_metrics['progress'])
        logger.info(f"\nObject-Avoidance Baseline:")
        logger.info(f"  Mean Progress: {mean_progress:.1f}%")
        logger.info(f"  Note: Agent not trained for obstacles - expect collisions")
    
    if head_to_bot_metrics:
        mean_progress = sum(head_to_bot_metrics['progress']) / len(head_to_bot_metrics['progress'])
        logger.info(f"\nHead-to-Bot Baseline:")
        logger.info(f"  Mean Progress: {mean_progress:.1f}%")
        logger.info(f"  Note: Agent not trained for bot cars - expect collisions")
    
    logger.info("\n" + "="*70)
    logger.info("Baseline evaluation complete!")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("  1. Train Object-Avoidance agent: python scripts/training/train_part2_object_avoidance.py")
    logger.info("  2. Train Head-to-Bot agent: python scripts/training/train_part2_head_to_bot.py")
    logger.info("  3. Compare Part II agents' performance against these baselines")

if __name__ == "__main__":
    main()

