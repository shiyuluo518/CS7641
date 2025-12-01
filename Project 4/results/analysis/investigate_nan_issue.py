"""
Investigate the NaN issue in model predictions.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from src.utils import make_environment
from src.agents import MyFancyAgent

def check_model_weights(model_path):
    """Check if model weights contain NaN values."""
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    print("\nChecking model weights for NaN values...")
    nan_found = False
    
    # Check policy network
    for name, param in model.policy.named_parameters():
        if torch.isnan(param).any():
            print(f"  [ERROR] NaN found in {name}")
            nan_found = True
        elif torch.isinf(param).any():
            print(f"  [ERROR] Inf found in {name}")
            nan_found = True
    
    if not nan_found:
        print("  [OK] No NaN or Inf values in model weights")
    
    # Check observation space
    print(f"\nObservation space: {model.observation_space}")
    print(f"Action space: {model.action_space}")
    
    return model, nan_found

def test_model_prediction(model, obs_shape=(38464,)):
    """Test model prediction with sample observations."""
    print(f"\nTesting model prediction with observation shape: {obs_shape}")
    
    # Create test observation
    test_obs = np.random.randn(*obs_shape).astype(np.float32)
    
    # Normalize to reasonable range
    test_obs = np.clip(test_obs, -1, 1)
    
    print(f"Test observation shape: {test_obs.shape}")
    print(f"Test observation range: [{test_obs.min():.3f}, {test_obs.max():.3f}]")
    print(f"Test observation has NaN: {np.isnan(test_obs).any()}")
    print(f"Test observation has Inf: {np.isinf(test_obs).any()}")
    
    try:
        action, _ = model.predict(test_obs, deterministic=True)
        print(f"  [OK] Prediction successful: action={action}")
        return True
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_env():
    """Test model with real environment."""
    print("\n" + "="*70)
    print("Testing with Real Environment")
    print("="*70)
    
    env = make_environment('deepracer-v0')
    agent = MyFancyAgent(environment=env, name='test_agent')
    
    model_path = 'models/time_trial_ppo_iteration1_baseline_final_1763757450.zip'
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    agent.load(model_path)
    
    print("\nTesting agent with real environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Observation has NaN: {np.isnan(obs).any()}")
    print(f"Observation has Inf: {np.isinf(obs).any()}")
    
    try:
        action = agent.get_action(obs, deterministic=True)
        print(f"  [OK] Action obtained: {action}")
    except Exception as e:
        print(f"  [ERROR] Failed to get action: {e}")
        import traceback
        traceback.print_exc()
    
    env.close()

if __name__ == "__main__":
    print("="*70)
    print("NaN Issue Investigation")
    print("="*70)
    
    model_path = 'models/time_trial_ppo_iteration1_baseline_final_1763757450.zip'
    
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    
    # Check model weights
    model, has_nan = check_model_weights(model_path)
    
    # Test prediction
    obs_shape = (38464,)  # LIDAR (64) + STEREO_CAMERAS (38400)
    test_model_prediction(model, obs_shape)
    
    # Test with real environment
    test_with_real_env()
    
    print("\n" + "="*70)
    print("Investigation Complete")
    print("="*70)
