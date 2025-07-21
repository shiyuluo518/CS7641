import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

def dqn_cartpole(num_episodes=500, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=500):
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQNNet(state_dim, action_dim)
    target_net = DQNNet(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start
    rewards = []
    steps_done = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            steps_done += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state)).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            # Learn
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards_b, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)
                q_values = policy_net(states).gather(1, actions)
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0].unsqueeze(1)
                    q_target = rewards_b + gamma * q_next * (1 - dones)
                loss = nn.MSELoss()(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update target network
            if steps_done % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())
        rewards.append(total_reward)
        epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay)
    env.close()
    return rewards

if __name__ == "__main__":
    rewards = dqn_cartpole(num_episodes=500)
    print(f"DQN CartPole average reward (last 100 episodes): {np.mean(rewards[-100:])}") 