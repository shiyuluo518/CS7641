"""
QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
Implementation based on Rashid et al. 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    Q-network for individual agents.
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64, n_layers=2):
        super(QNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.q_network = nn.Sequential(*layers)
    
    def forward(self, obs):
        """
        Forward pass through Q-network.
        
        Args:
            obs: Observation tensor (batch_size, obs_dim)
            
        Returns:
            Q-values for all actions (batch_size, action_dim)
        """
        return self.q_network(obs)


class MixingNetwork(nn.Module):
    """
    Mixing network that combines individual Q-values into joint Q-value.
    Ensures monotonicity: ∂Q_tot / ∂Q_a >= 0 for all agents a.
    """
    
    def __init__(self, n_agents, state_dim, hidden_dim=64, hypernet_hidden_dim=64):
        super(MixingNetwork, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetworks for generating weights
        # First layer weights: (batch, n_agents, hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, n_agents * hidden_dim)
        )
        
        # Second layer weights: (batch, hidden_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim)
        )
        
        # Hypernetwork for bias
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim)
        )
        
        # Second bias is a scalar
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1)
        )
    
    def forward(self, q_values, states):
        """
        Mix individual Q-values into joint Q-value.
        
        Args:
            q_values: Individual Q-values (batch_size, n_agents)
            states: Global state (batch_size, state_dim)
            
        Returns:
            Joint Q-value (batch_size, 1)
        """
        batch_size = q_values.size(0)
        
        # Generate weights from hypernetworks
        w1 = torch.abs(self.hyper_w1(states))  # Abs ensures monotonicity
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        b1 = self.hyper_b1(states)
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        
        b2 = self.hyper_b2(states)
        b2 = b2.view(batch_size, 1, 1)  # Reshape to match bmm output shape
        
        # First layer
        q_values = q_values.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)
        
        # Second layer
        q_total = torch.bmm(hidden, w2) + b2
        
        return q_total.view(batch_size, 1)


class QMIX:
    """
    QMIX algorithm for multi-agent reinforcement learning.
    """
    
    def __init__(self, obs_dim, action_dim, state_dim, n_agents=2,
                 hidden_dim=64, lr=5e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=50000, target_update_interval=200,
                 device='cpu'):
        """
        Initialize QMIX.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            state_dim: Global state dimension (for mixing network)
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
        self.state_dim = state_dim
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
            hypernet_dim = config_module.HYPERNET_HIDDEN_DIM
        except:
            n_layers = 2  # Default fallback
            hypernet_dim = 64  # Default fallback
        
        # Q-networks for each agent (use config parameters for layers)
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
        
        # Mixing networks (use config parameters for hypernetwork)
        self.mixing_network = MixingNetwork(n_agents, state_dim, hidden_dim, 
                                           hypernet_hidden_dim=hypernet_dim).to(device)
        self.target_mixing_network = MixingNetwork(n_agents, state_dim, hidden_dim,
                                                   hypernet_hidden_dim=hypernet_dim).to(device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Optimizer
        params = list(self.q_networks.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
        # Set target networks to eval mode
        for net in self.target_q_networks:
            net.eval()
        self.target_mixing_network.eval()
    
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
    
    def update(self, batch, states, next_states):
        """
        Update Q-networks and mixing network using a batch of experiences.
        
        Args:
            batch: Dictionary containing:
                - obs: Observations (batch_size, n_agents, obs_dim) - should already be on device
                - actions: Actions (batch_size, n_agents)
                - rewards: Rewards (batch_size, n_agents)
                - next_obs: Next observations (batch_size, n_agents, obs_dim)
                - dones: Done flags (batch_size,)
            states: Global states (batch_size, state_dim) - should already be on device
            next_states: Next global states (batch_size, state_dim) - should already be on device
        """
        # Batch and states should already be on device from replay buffer
        # Only move if they're not already on the correct device (safety check)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) and v.device != self.device else v 
                 for k, v in batch.items()}
        if isinstance(states, torch.Tensor) and states.device != self.device:
            states = states.to(self.device)
        if isinstance(next_states, torch.Tensor) and next_states.device != self.device:
            next_states = next_states.to(self.device)
        
        batch_size = batch['obs'].size(0)
        
        # Compute current Q-values
        q_values = []
        for i in range(self.n_agents):
            agent_obs = batch['obs'][:, i, :]
            agent_actions = batch['actions'][:, i]
            q_i = self.q_networks[i](agent_obs)
            q_i = q_i.gather(1, agent_actions.unsqueeze(1))
            q_values.append(q_i)
        
        q_values = torch.cat(q_values, dim=1)  # (batch_size, n_agents)
        q_total = self.mixing_network(q_values, states)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(self.n_agents):
                next_agent_obs = batch['next_obs'][:, i, :]
                next_q_i = self.target_q_networks[i](next_agent_obs)
                next_q_i = next_q_i.max(1)[0].unsqueeze(1)
                next_q_values.append(next_q_i)
            
            next_q_values = torch.cat(next_q_values, dim=1)  # (batch_size, n_agents)
            next_q_total = self.target_mixing_network(next_q_values, next_states)
            
            # Compute total reward (sum of individual rewards)
            total_rewards = batch['rewards'].sum(dim=1, keepdim=True)
            target_q = total_rewards + self.gamma * next_q_total * (1 - batch['dones'].float().unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(q_total, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_networks.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update epsilon
        self.steps += 1
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_decay)
        
        # Update target networks
        if self.steps % self.target_update_interval == 0:
            for i in range(self.n_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        return loss.item()
    
    def save(self, filepath):
        """Save model to file."""
        torch.save({
            'q_networks': self.q_networks.state_dict(),
            'target_q_networks': self.target_q_networks.state_dict(),
            'mixing_network': self.mixing_network.state_dict(),
            'target_mixing_network': self.target_mixing_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon,
        }, filepath)
    
    def load(self, filepath):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_networks.load_state_dict(checkpoint['q_networks'])
        self.target_q_networks.load_state_dict(checkpoint['target_q_networks'])
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']

