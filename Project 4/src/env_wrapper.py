"""
Environment wrapper for stable-baselines3 compatibility.
Handles the flattened observation space and converts it for PPO.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Union

from src.transforms import (
    UnflattenObservation,
    LIDAR_FLATTEN_SHAPE,
    STEREO_FLATTEN_CAMERA_SHAPE
)


class DeepRacerFeatureExtractor:
    """
    Custom feature extractor for stable-baselines3 that handles
    flattened observations from DeepRacer environment.
    """
    def __init__(self, observation_space: spaces.Box):
        self.observation_space = observation_space
        self.unflatten = UnflattenObservation()
        
        # Calculate the flattened observation size
        # LIDAR: 64, STEREO_CAMERAS: 2 * 120 * 160 = 38400
        self.flat_obs_size = LIDAR_FLATTEN_SHAPE + STEREO_FLATTEN_CAMERA_SHAPE
        self.feature_dim = observation_space.shape[0]  # Use original observation size
        
    def extract_features(self, observation: np.ndarray) -> np.ndarray:
        """
        Extract features from flattened observation.
        Normalizes observations to help with training stability.
        """
        # Ensure observation is the right shape and type
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        normalized = observation.copy().astype(np.float32)
        
        # Normalize camera portion (last 38400 values) to [0, 1]
        camera_start = LIDAR_FLATTEN_SHAPE
        camera_end = camera_start + STEREO_FLATTEN_CAMERA_SHAPE
        if normalized.shape[1] >= camera_end:
            # Normalize camera values from [0, 255] to [0, 1]
            normalized[:, camera_start:camera_end] = np.clip(
                normalized[:, camera_start:camera_end] / 255.0, 
                0.0, 1.0
            )
        
        # Normalize LIDAR portion (first 64 values)
        lidar_end = min(LIDAR_FLATTEN_SHAPE, normalized.shape[1])
        if lidar_end > 0:
            # Handle Inf values in LIDAR first
            lidar_data = normalized[:, :lidar_end].copy()
            lidar_data = np.where(np.isinf(lidar_data), 1.0, lidar_data)  # Replace Inf with max range
            # Clip LIDAR to reasonable range [0.15, 1.0] and normalize to [0, 1]
            lidar_data = np.clip(lidar_data, 0.15, 1.0)
            normalized[:, :lidar_end] = (lidar_data - 0.15) / (1.0 - 0.15)
        
        # Handle Inf values in camera data
        camera_start = LIDAR_FLATTEN_SHAPE
        camera_end = camera_start + STEREO_FLATTEN_CAMERA_SHAPE
        if normalized.shape[1] >= camera_end:
            camera_data = normalized[:, camera_start:camera_end]
            camera_data = np.where(np.isinf(camera_data), 255.0, camera_data)  # Replace Inf with max
            normalized[:, camera_start:camera_end] = camera_data
        
        # Ensure no NaN or Inf values - final cleanup
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Final clip to prevent extreme values
        normalized = np.clip(normalized, -10.0, 10.0)
        
        return normalized.flatten() if normalized.shape[0] == 1 else normalized


class DeepRacerWrapper(gym.ObservationWrapper):
    """
    Wrapper that ensures observations are properly formatted.
    The environment already uses FlattenObservation, so this
    mainly handles normalization and feature extraction.
    """
    def __init__(self, env):
        super().__init__(env)
        self.feature_extractor = DeepRacerFeatureExtractor(env.observation_space)
        
        # Update observation space to match extracted features
        # For simplicity, we'll keep the same space but ensure it's Box
        if isinstance(env.observation_space, spaces.Box):
            self.observation_space = env.observation_space
    
    def observation(self, obs):
        """Process observation through feature extractor."""
        return self.feature_extractor.extract_features(obs)

