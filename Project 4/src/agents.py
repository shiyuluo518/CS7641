import abc
import torch
import torch.nn as nn

from src.transforms import EncodeObservation


class Agent(nn.Module, abc.ABC):
    '''Boilerplate class for providing interface'''
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    @abc.abstractmethod
    def get_action(self, x):
        raise NotImplementedError


class RandomAgent(Agent):
    '''
    A random agent for demonstrating usage of the environment
    '''
    def __init__(self, environment, name='random'):
        super().__init__(name=name)
        self.action_space = environment.action_space        

    def get_action(self, x):
        return self.action_space.sample()


class MyFancyAgent(Agent):
    '''
    PPO-based DeepRacer agent using stable-baselines3.
    '''
    def __init__(self, environment, name='my_fancy_agent', **ppo_kwargs):
        super().__init__(name=name)
        
        # Import PPOAgent here to avoid circular imports
        from src.ppo_agent import PPOAgent
        
        # Create PPO agent with default or custom parameters
        self.ppo_agent = PPOAgent(
            environment=environment,
            name=name,
            **ppo_kwargs
        )
        
        # Store environment and action space for compatibility
        self.environment = environment
        self.action_space = environment.action_space
    
    def get_action(self, x, deterministic=False):
        """
        Get action from PPO policy.
        
        Args:
            x: Observation (flattened array)
            deterministic: If True, use deterministic policy
        
        Returns:
            Action
        """
        return self.ppo_agent.get_action(x, deterministic=deterministic)
    
    def train(self, total_timesteps, **kwargs):
        """Train the PPO agent."""
        return self.ppo_agent.train(total_timesteps, **kwargs)
    
    def save(self, path):
        """Save the model."""
        self.ppo_agent.save(path)
    
    def load(self, path):
        """Load a saved model."""
        self.ppo_agent.load(path)
    
    def evaluate(self, n_episodes=10, deterministic=True):
        """Evaluate the agent."""
        return self.ppo_agent.evaluate(n_episodes, deterministic)