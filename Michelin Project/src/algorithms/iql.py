"""
Independent Q-Learning (IQL): Baseline multi-agent RL algorithm.
Each agent learns its own Q-function independently, without coordination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .qmix import QNetwork


class IQL:
    """
    Independent Q-Learning algorithm for multi-agent reinforcement learning.
    Each agent learns its Q-function independently without any mixing or coordination.
    """
    
    def __init__(self, obs_dim, action_dim, n_agents=2,
                 hidden_dim=64, lr=5e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=50000, target_update_interval=200,
                 device='cpu'):
        """
        Initialize IQL.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            n_agents: Number of agents
            hidden_dim: Hidden dimension for networks
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay rate
            target_update_interval: Steps between target network updates
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_interval = target_update_interval
        self.device = device
        self.steps = 0
        
        # Import config for network architecture parameters
        try:
            import src.config as config_module
            n_layers = config_module.N_LAYERS
        except:
            n_layers = 2  # Default fallback
        
        # Q-networks for each agent (each learns independently, use config parameters)
        self.q_networks = nn.ModuleList([
            QNetwork(obs_dim, action_dim, hidden_dim, n_layers=n_layers).to(device)
            for _ in range(n_agents)
        ])
        
        # Target Q-networks
        self.target_q_networks = nn.ModuleList([
            QNetwork(obs_dim, action_dim, hidden_dim, n_layers=n_layers).to(device)
            for _ in range(n_agents)
        ])
        
        # Copy weights to target networks
        for i in range(n_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        # Optimizers (one per agent)
        self.optimizers = [
            torch.optim.Adam(self.q_networks[i].parameters(), lr=lr)
            for i in range(n_agents)
        ]
        
        # Set target networks to eval mode
        for net in self.target_q_networks:
            net.eval()
    
    def select_actions(self, obs_list, training=True):
        """
        Select actions for all agents using epsilon-greedy policy.
        Optimized for GPU: batch action selection when possible.
        
        Args:
            obs_list: List of observations for each agent
            training: Whether in training mode (affects exploration)
            
        Returns:
            List of actions for each agent
        """
        # Batch action selection on GPU for faster computation
        if not training or np.random.random() >= self.epsilon:
            # Greedy actions: batch compute all agents on GPU simultaneously
            obs_tensor = torch.stack([torch.tensor(obs, dtype=torch.float32, device=self.device) 
                                    for obs in obs_list], dim=0)  # Shape: (n_agents, obs_dim)
            with torch.no_grad():
                # Compute Q-values for all agents in parallel (much faster on GPU)
                q_values_list = [self.q_networks[i](obs_tensor[i:i+1]) for i in range(len(obs_list))]
                q_values_batch = torch.cat(q_values_list, dim=0)  # Shape: (n_agents, action_dim)
                actions = q_values_batch.argmax(dim=-1).cpu().numpy().tolist()
        else:
            # Random actions (exploration)
            actions = [np.random.randint(0, self.action_dim) for _ in obs_list]
        
        return actions
    
    def update(self, batch):
        """
        Update Q-networks using a batch of experiences.
        Each agent updates independently based on its own observations and rewards.
        
        Args:
            batch: Dictionary containing:
                - obs: Observations (batch_size, n_agents, obs_dim) - should already be on device
                - actions: Actions (batch_size, n_agents)
                - rewards: Rewards (batch_size, n_agents)
                - next_obs: Next observations (batch_size, n_agents, obs_dim)
                - dones: Done flags (batch_size,)
        """
        # Batch should already be on device from replay buffer
        # Only move if not already on the correct device (safety check)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) and v.device != self.device else v 
                 for k, v in batch.items()}
        
        batch_size = batch['obs'].size(0)
        total_loss = 0.0
        
        # Update each agent independently
        for i in range(self.n_agents):
            # Get agent-specific observations, actions, and rewards
            agent_obs = batch['obs'][:, i, :]
            agent_actions = batch['actions'][:, i]
            agent_rewards = batch['rewards'][:, i]
            agent_next_obs = batch['next_obs'][:, i, :]
            dones = batch['dones']
            
            # Compute current Q-values
            q_values = self.q_networks[i](agent_obs)
            q_i = q_values.gather(1, agent_actions.unsqueeze(1))
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_q_networks[i](agent_next_obs)
                next_q_i = next_q_values.max(1)[0].unsqueeze(1)
                target_q = agent_rewards.unsqueeze(1) + self.gamma * next_q_i * (1 - dones.float().unsqueeze(1))
            
            # Compute loss
            loss = F.mse_loss(q_i, target_q)
            
            # Optimize
            self.optimizers[i].zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), 10.0)
            self.optimizers[i].step()
            
            total_loss += loss.item()
        
        # Update epsilon
        self.steps += 1
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_decay)
        
        # Update target networks
        if self.steps % self.target_update_interval == 0:
            for i in range(self.n_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        return total_loss / self.n_agents  # Average loss across agents
    
    def save(self, filepath):
        """Save model to file."""
        torch.save({
            'q_networks': self.q_networks.state_dict(),
            'target_q_networks': self.target_q_networks.state_dict(),
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'steps': self.steps,
            'epsilon': self.epsilon,
        }, filepath)
    
    def load(self, filepath):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_networks.load_state_dict(checkpoint['q_networks'])
        self.target_q_networks.load_state_dict(checkpoint['target_q_networks'])
        for i, opt_state in enumerate(checkpoint['optimizers']):
            self.optimizers[i].load_state_dict(opt_state)
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']

