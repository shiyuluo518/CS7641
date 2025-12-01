"""
PPO Agent implementation using stable-baselines3 with enhanced metrics tracking.
"""
import os
import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.agents import Agent
from src.env_wrapper import DeepRacerWrapper


class EnhancedProgressCallback(BaseCallback):
    """
    Enhanced callback for logging 3 key metrics:
    1. Episode Reward
    2. Progress Percentage
    3. Lap Time (or episode length as proxy)
    """
    
    def __init__(self, verbose=1, log_interval=10):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_progress = []  # Track progress percentage
        self.last_log_time = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Log custom metrics if available
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                reward = info['episode']['r']
                length = info['episode']['l']
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.episode_count += 1
                
                # Extract progress from info if available
                progress = 0.0
                if 'reward_params' in info:
                    progress = info.get('reward_params', {}).get('progress', 0.0)
                elif 'progress' in info:
                    progress = info['progress']
                self.episode_progress.append(progress)
                
                # Log to TensorBoard - 3 key metrics
                # Note: self.logger.record() automatically uses self.num_timesteps as the step
                # This ensures metrics are plotted against global steps, not episode numbers
                self.logger.record('train/episode_reward', reward)
                self.logger.record('train/episode_length', length)
                self.logger.record('train/episode_progress', progress)
                
                # Calculate lap time proxy (steps to complete, lower is better)
                # For Time-Trial, we want to minimize lap time
                lap_time_proxy = length  # Use episode length as proxy for lap time
                self.logger.record('train/lap_time_proxy', lap_time_proxy)
                
                # Also store step number for each episode for plotting
                if not hasattr(self, 'episode_steps'):
                    self.episode_steps = []
                self.episode_steps.append(self.num_timesteps)
                
                # Print progress to console
                if self.verbose >= 1:
                    mean_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                    mean_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
                    mean_progress = np.mean(self.episode_progress[-10:]) if len(self.episode_progress) >= 10 else np.mean(self.episode_progress)
                    
                    print(f"\n{'='*60}")
                    print(f"Episode {self.episode_count} | Step {self.num_timesteps}")
                    print(f"  Reward: {reward:.2f} | Length: {length} | Progress: {progress:.1f}%")
                    print(f"  Mean (last 10): Reward={mean_reward:.2f}, Length={mean_length:.1f}, Progress={mean_progress:.1f}%")
                    print(f"  Progress: {100*self.num_timesteps/self.locals.get('total_timesteps', 1):.1f}%")
                    print(f"{'='*60}")
        
        # Periodic summary
        if self.num_timesteps - self.last_log_time >= self.log_interval * 1000:
            if len(self.episode_rewards) > 0:
                print(f"\n[Training Summary] (up to step {self.num_timesteps}):")
                print(f"  Episodes: {self.episode_count}")
                print(f"  Mean Reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
                print(f"  Mean Length: {np.mean(self.episode_lengths):.1f} ± {np.std(self.episode_lengths):.1f}")
                print(f"  Mean Progress: {np.mean(self.episode_progress):.1f}% ± {np.std(self.episode_progress):.1f}%")
                print(f"  Best Reward: {np.max(self.episode_rewards):.2f}")
                print(f"  Best Progress: {np.max(self.episode_progress):.1f}%")
                print(f"  Recent (last 10) Mean: Reward={np.mean(self.episode_rewards[-10:]):.2f}, Progress={np.mean(self.episode_progress[-10:]):.1f}%")
            self.last_log_time = self.num_timesteps
        
        return True


class PPOAgent(Agent):
    """
    PPO Agent using stable-baselines3.
    Wraps the PPO model to match the Agent interface.
    """
    
    def __init__(
        self,
        environment,
        name='ppo_agent',
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[dict] = None,
        device: Union[str, torch.device] = 'auto',
        verbose: int = 1,
        use_cnn: bool = False,  # Option to use CNN instead of MLP
        **kwargs
    ):
        super().__init__(name=name)
        
        self.environment = environment
        self.action_space = environment.action_space
        
        # Wrap environment for stable-baselines3
        wrapped_env = DeepRacerWrapper(environment)
        wrapped_env = Monitor(wrapped_env)
        
        # Create a dummy vec env (required by SB3, but only 1 env)
        self.vec_env = DummyVecEnv([lambda: wrapped_env])
        
        # Default policy kwargs
        if policy_kwargs is None:
            if use_cnn:
                # Use CNN-based policy (requires custom feature extractor)
                # For now, we'll use MLP but with better architecture
                policy_kwargs = {
                    'net_arch': dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                    'activation_fn': torch.nn.ReLU,
                }
            else:
                # Use MLP with improved architecture
                policy_kwargs = {
                    'net_arch': dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                    'activation_fn': torch.nn.ReLU,
                }
        
        # Create PPO model
        self.model = PPO(
            policy='MlpPolicy',  # Using MLP for flattened observations
            env=self.vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log="runs",
            verbose=verbose,
            device=device,
            policy_kwargs=policy_kwargs,
            **kwargs
        )
        
        self.trained = False
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Get action from the policy.
        
        Args:
            observation: Flattened observation array
            deterministic: If True, use deterministic policy (no exploration)
        
        Returns:
            Action (int for discrete action space)
        """
        if not self.trained:
            # If not trained, return random action
            return self.action_space.sample()
        
        try:
            # Convert to numpy if it's a tensor (handle BEFORE copy())
            if torch.is_tensor(observation):
                observation = observation.cpu().detach().numpy()
            elif hasattr(observation, 'cpu'):
                observation = observation.cpu().numpy()
            elif hasattr(observation, 'numpy'):
                observation = observation.numpy()
            
            # Ensure it's a numpy array before copying
            observation = np.asarray(observation, dtype=np.float32)
            
            # Clean observation: handle NaN and Inf values
            observation = observation.copy()
            
            # Replace Inf with large finite values
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Clip to reasonable range to prevent extreme values
            observation = np.clip(observation, -10.0, 10.0)
            
            # Ensure observation is the right shape
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            
            # Get action from model
            action, _ = self.model.predict(observation, deterministic=deterministic)
            
            # Handle single action
            if isinstance(action, np.ndarray) and action.size == 1:
                return int(action.item())
            return int(action)
        except Exception as e:
            # Fallback to random if prediction fails
            print(f"Warning: Model prediction failed ({e}), using random action")
            return self.action_space.sample()
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        save_path: Optional[str] = None,
        callback: Optional[BaseCallback] = None,
        verbose: int = 1
    ):
        """
        Train the PPO agent with enhanced metrics tracking.
        
        Args:
            total_timesteps: Total number of timesteps to train
            log_interval: Logging interval for stable-baselines3
            save_path: Path to save checkpoints (optional)
            callback: Custom callback (optional)
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        callbacks = []
        
        # Add checkpoint callback if save_path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=max(total_timesteps // 10, 1000),
                save_path=os.path.dirname(save_path),
                name_prefix=os.path.basename(save_path).replace('.zip', '')
            )
            callbacks.append(checkpoint_callback)
        
        # Add enhanced progress callback with 3 metrics
        progress_callback = EnhancedProgressCallback(verbose=verbose, log_interval=log_interval)
        callbacks.append(progress_callback)
        
        # Add custom callback if provided
        if callback:
            callbacks.append(callback)
        
        print(f"\n{'='*70}")
        print(f"Starting PPO Training")
        print(f"{'='*70}")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Learning Rate: {self.model.learning_rate}")
        print(f"N Steps: {self.model.n_steps}")
        print(f"Batch Size: {self.model.batch_size}")
        print(f"N Epochs: {self.model.n_epochs}")
        print(f"Gamma: {self.model.gamma}")
        print(f"Tracking Metrics: Episode Reward, Progress %, Lap Time Proxy")
        print(f"{'='*70}\n")
        
        # Train the model
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                log_interval=log_interval,
                callback=callbacks if callbacks else None,
                progress_bar=True
            )
        except ImportError:
            # Fallback if progress bar dependencies not available
            self.model.learn(
                total_timesteps=total_timesteps,
                log_interval=log_interval,
                callback=callbacks if callbacks else None,
                progress_bar=False
            )
        
        self.trained = True
        
        # Print final summary with 3 metrics
        if len(progress_callback.episode_rewards) > 0:
            print(f"\n{'='*70}")
            print(f"Training Completed!")
            print(f"{'='*70}")
            print(f"Total Episodes: {progress_callback.episode_count}")
            print(f"Final Statistics:")
            print(f"  Mean Reward: {np.mean(progress_callback.episode_rewards):.2f} ± {np.std(progress_callback.episode_rewards):.2f}")
            print(f"  Mean Length: {np.mean(progress_callback.episode_lengths):.1f} ± {np.std(progress_callback.episode_lengths):.1f}")
            print(f"  Mean Progress: {np.mean(progress_callback.episode_progress):.1f}% ± {np.std(progress_callback.episode_progress):.1f}%")
            print(f"  Best Reward: {np.max(progress_callback.episode_rewards):.2f}")
            print(f"  Best Progress: {np.max(progress_callback.episode_progress):.1f}%")
            print(f"  Recent (last 10) Mean: Reward={np.mean(progress_callback.episode_rewards[-10:]):.2f}, Progress={np.mean(progress_callback.episode_progress[-10:]):.1f}%")
            print(f"{'='*70}\n")
            
            # Save training metrics to file for plotting
            import json
            training_data_path = 'results/training_metrics.json'
            os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
            training_metrics = {
                'episode_rewards': progress_callback.episode_rewards,
                'episode_lengths': progress_callback.episode_lengths,
                'episode_progress': progress_callback.episode_progress,
                'episode_steps': [progress_callback.log_interval * 1000 * (i // (progress_callback.log_interval * 100)) for i in range(len(progress_callback.episode_rewards))] if hasattr(progress_callback, 'log_interval') else list(range(len(progress_callback.episode_rewards)))
            }
            with open(training_data_path, 'w') as f:
                json.dump(training_metrics, f, indent=2)
            print(f"[SAVED] Training metrics saved to: {training_data_path}")
        
        # Save final model
        if save_path:
            self.model.save(save_path)
            print(f"[SAVED] Model saved to: {save_path}")
    
    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a saved model."""
        self.model = PPO.load(path, env=self.vec_env)
        self.trained = True
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to run for evaluation.
            deterministic: Whether to use deterministic actions for evaluation.
        
        Returns:
            A dictionary containing mean reward, std reward, mean length, and std length.
        """
        rewards = []
        lengths = []
        progress_values = []
        
        for episode_idx in range(n_episodes):
            try:
                obs = self.vec_env.reset()
                # Handle both single obs and (obs, info) tuple
                if isinstance(obs, tuple):
                    obs, info = obs
                else:
                    info = [{}] if isinstance(obs, np.ndarray) else {}
                
                # Ensure obs is numpy array
                if isinstance(obs, np.ndarray) and obs.ndim == 1:
                    obs = obs.reshape(1, -1)
                
                done = False
                episode_reward = 0
                episode_length = 0
                episode_progress = 0.0
                max_steps = 1000  # Safety limit
                
                step_count = 0
                while not done and step_count < max_steps:
                    try:
                        action, _ = self.model.predict(obs, deterministic=deterministic)
                        obs, reward, done, info = self.vec_env.step(action)
                        
                        # Handle reward (could be array or scalar)
                        if isinstance(reward, np.ndarray):
                            episode_reward += float(reward[0])
                        else:
                            episode_reward += float(reward)
                        
                        episode_length += 1
                        step_count += 1
                        
                        # Extract progress if available
                        if isinstance(info, list) and len(info) > 0:
                            if 'reward_params' in info[0]:
                                episode_progress = max(episode_progress, info[0].get('reward_params', {}).get('progress', 0.0))
                        elif isinstance(info, dict) and 'reward_params' in info:
                            episode_progress = max(episode_progress, info.get('reward_params', {}).get('progress', 0.0))
                        
                        # Handle done (could be array or scalar)
                        if isinstance(done, np.ndarray):
                            if done[0]:
                                break
                        elif done:
                            break
                            
                    except Exception as e:
                        print(f"Warning: Error during episode {episode_idx} step {step_count}: {e}")
                        break
            except Exception as e:
                print(f"Warning: Error resetting environment for episode {episode_idx}: {e}")
                # Use default values for failed episode
                episode_reward = 0.0
                episode_length = 0
                episode_progress = 0.0
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            progress_values.append(episode_progress)
        
        # Calculate statistics, handling empty lists
        if len(rewards) == 0:
            return {
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "mean_length": 0.0,
                "std_length": 0.0,
                "mean_progress": 0.0,
                "std_progress": 0.0,
            }
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "mean_progress": float(np.mean(progress_values)),
            "std_progress": float(np.std(progress_values)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_progress": float(np.min(progress_values)),
            "max_progress": float(np.max(progress_values)),
            "completion_rate": float(np.sum(np.array(progress_values) >= 100.0) / len(progress_values) * 100.0),
        }
