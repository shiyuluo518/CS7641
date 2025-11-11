"""
Experience replay buffer for multi-agent reinforcement learning.
"""

import numpy as np
import torch
from collections import deque


class MultiAgentReplayBuffer:
    """
    Experience replay buffer for storing and sampling multi-agent transitions.
    Optimized for GPU acceleration.
    """
    
    def __init__(self, capacity=100000, state_dim=None, device='cpu'):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Fixed state dimension (if None, will be inferred from first sample)
            device: Device to create tensors on ('cpu' or 'cuda')
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.device = device
    
    def push(self, obs, actions, rewards, next_obs, dones, state, next_state):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observations for all agents (list of arrays)
            actions: Actions for all agents (list of ints)
            rewards: Rewards for all agents (list of floats)
            next_obs: Next observations for all agents (list of arrays)
            dones: Done flags (bool or list)
            state: Global state (array)
            next_state: Next global state (array)
        """
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones,
            'state': state,
            'next_state': next_state
        })
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary of batched transitions
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        # Extract and stack
        n_agents = len(batch[0]['obs'])
        obs_dim = len(batch[0]['obs'][0])
        
        # Use fixed state_dim if provided, otherwise infer from batch
        if self.state_dim is None:
            # Infer from first transition
            state_dim = len(batch[0]['state'])
        else:
            state_dim = self.state_dim
        
        # Create tensors directly on target device for faster GPU operations
        obs_batch = torch.zeros(batch_size, n_agents, obs_dim, device=self.device)
        actions_batch = torch.zeros(batch_size, n_agents, dtype=torch.long, device=self.device)
        rewards_batch = torch.zeros(batch_size, n_agents, device=self.device)
        next_obs_batch = torch.zeros(batch_size, n_agents, obs_dim, device=self.device)
        dones_batch = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        states_batch = torch.zeros(batch_size, state_dim, device=self.device)
        next_states_batch = torch.zeros(batch_size, state_dim, device=self.device)
        
        for i, transition in enumerate(batch):
            for j in range(n_agents):
                # Convert to tensor and move to device in one step
                obs_batch[i, j] = torch.tensor(transition['obs'][j], dtype=torch.float32, device=self.device)
                actions_batch[i, j] = transition['actions'][j]
                rewards_batch[i, j] = transition['rewards'][j]
                next_obs_batch[i, j] = torch.tensor(transition['next_obs'][j], dtype=torch.float32, device=self.device)
            
            dones_batch[i] = transition['dones'] if isinstance(transition['dones'], bool) else transition['dones'][0]
            
            # Handle state dimension mismatch - pad or truncate to match fixed state_dim
            state_array = np.array(transition['state'], dtype=np.float32)
            next_state_array = np.array(transition['next_state'], dtype=np.float32)
            
            # Pad or truncate states to match state_dim
            if len(state_array) < state_dim:
                # Pad with zeros
                padding = np.zeros(state_dim - len(state_array), dtype=np.float32)
                state_array = np.concatenate([state_array, padding])
            elif len(state_array) > state_dim:
                # Truncate
                state_array = state_array[:state_dim]
            
            if len(next_state_array) < state_dim:
                # Pad with zeros
                padding = np.zeros(state_dim - len(next_state_array), dtype=np.float32)
                next_state_array = np.concatenate([next_state_array, padding])
            elif len(next_state_array) > state_dim:
                # Truncate
                next_state_array = next_state_array[:state_dim]
            
            # Create tensors directly on device
            states_batch[i] = torch.tensor(state_array, dtype=torch.float32, device=self.device)
            next_states_batch[i] = torch.tensor(next_state_array, dtype=torch.float32, device=self.device)
        
        return {
            'obs': obs_batch,
            'actions': actions_batch,
            'rewards': rewards_batch,
            'next_obs': next_obs_batch,
            'dones': dones_batch,
            'states': states_batch,
            'next_states': next_states_batch
        }
    
    def __len__(self):
        return len(self.buffer)
