"""
Generate demo videos for all 3 race types: Time-Trial, Object-Avoidance, and Head-to-Bot.

Usage:
    python generate_demo_videos.py
"""
import os
import yaml
from pathlib import Path
from loguru import logger
from src.utils import demo, make_environment
from src.agents import MyFancyAgent

# Ensure we're in project root
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

ENVIRONMENT_PARAMS_PATH = 'configs/environment_params.yaml'
MODELS_DIR = 'models'
RESULTS_DEMOS_DIR = 'results/demos'

def find_latest_model():
    """Find the latest trained model."""
    model_files = list(Path(MODELS_DIR).glob("*.zip"))
    if not model_files:
        raise FileNotFoundError("No model files found in models/ directory")
    
    # Get the most recent model (prefer final model)
    final_models = [m for m in model_files if 'final' in m.name]
    if final_models:
        latest_model = max(final_models, key=lambda p: p.stat().st_mtime)
    else:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"Using model: {latest_model.name}")
    return str(latest_model)

def update_config(world_name, obstacles, bot_cars):
    """Update environment configuration."""
    with open(ENVIRONMENT_PARAMS_PATH, 'r') as f:
        env_params = yaml.safe_load(f)
    
    env_params['WORLD_NAME'] = world_name
    env_params['NUMBER_OF_OBSTACLES'] = obstacles
    env_params['NUMBER_OF_BOT_CARS'] = bot_cars
    
    with open(ENVIRONMENT_PARAMS_PATH, 'w') as f:
        yaml.dump(env_params, f, default_flow_style=False)
    
    logger.info(f"Updated config: WORLD_NAME={world_name}, OBSTACLES={obstacles}, BOT_CARS={bot_cars}")

def generate_video(model_path, race_type, track_name='reInvent2019_wide', obstacles=0, bot_cars=0):
    """Generate demo video for a specific race type."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Generating {race_type} Demo Video on {track_name}")
    logger.info(f"{'='*70}")
    
    # Update config
    update_config(track_name, obstacles, bot_cars)
    
    if obstacles > 0 or bot_cars > 0:
        logger.warning("Note: You may need to restart the simulation container for changes to take effect.")
        logger.info("Continuing with video generation (non-interactive mode)...")
    
    # Create environment and agent
    env = make_environment('deepracer-v0')
    agent = MyFancyAgent(environment=env, name=f'{race_type}_agent')
    
    # Load model
    agent.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Generate demo
    try:
        demo(agent, environment_name='deepracer-v0', directory=RESULTS_DEMOS_DIR)
        logger.info(f"{race_type} video generated successfully!")
    except Exception as e:
        logger.error(f"Error generating {race_type} video: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()

def main():
    """Generate all 3 demo videos."""
    logger.info("="*70)
    logger.info("Demo Video Generation for All Race Types")
    logger.info("="*70)
    
    # Find latest model
    try:
        model_path = find_latest_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please train a model first using: python train_part1_time_trial.py")
        return
    
    # Choose track
    track_name = 'reInvent2019_wide'
    logger.info(f"Using track: {track_name}")
    
    # Generate videos
    try:
        # 1. Time-Trial
        generate_video(model_path, 'time_trial', track_name, obstacles=0, bot_cars=0)
        
        # 2. Object-Avoidance
        generate_video(model_path, 'obstacle_avoidance', track_name, obstacles=6, bot_cars=0)
        
        # 3. Head-to-Bot
        generate_video(model_path, 'head_to_bot', track_name, obstacles=0, bot_cars=3)
        
        logger.info("\n" + "="*70)
        logger.info("Demo Video Generation Complete!")
        logger.info("="*70)
        logger.info(f"Videos saved to: {RESULTS_DEMOS_DIR}/")
        logger.info("\nGenerated videos:")
        logger.info("  1. Time-Trial")
        logger.info("  2. Object-Avoidance")
        logger.info("  3. Head-to-Bot")
        logger.info("\nVideos are ready to be included in README.md")
        
    except Exception as e:
        logger.error(f"Error generating videos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
