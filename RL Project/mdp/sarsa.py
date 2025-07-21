import gymnasium as gym
import numpy as np
from collections import defaultdict
from mdp.utils import discretize_state, epsilon_greedy

def sarsa(env, num_episodes=50000, alpha=0.1, gamma=1.0, epsilon=0.1, discretizer=None, bins=None, 
           epsilon_decay=None, epsilon_min=0.05):
    Q = defaultdict(float)
    nA = env.action_space.n
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Calculate current epsilon (decaying if specified)
        if epsilon_decay:
            current_epsilon = max(epsilon_min, epsilon * (epsilon_decay ** episode))
        else:
            current_epsilon = epsilon
            
        state, _ = env.reset()
        if discretizer:
            state = discretizer(state, bins)
        action = epsilon_greedy(Q, state, nA, current_epsilon)
        total_reward = 0
        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if discretizer:
                next_state = discretizer(next_state, bins)
            next_action = epsilon_greedy(Q, next_state, nA, current_epsilon)
            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])
            state, action = next_state, next_action
            total_reward += reward
        episode_rewards.append(total_reward)
    return Q, episode_rewards

def sarsa_with_decaying_epsilon(env, num_episodes=50000, alpha=0.1, gamma=1.0, epsilon_start=1.0, 
                                epsilon_min=0.05, discretizer=None, bins=None):
    """SARSA with decaying epsilon from epsilon_start to epsilon_min"""
    # Calculate decay rate to reach epsilon_min by the end
    epsilon_decay = (epsilon_min / epsilon_start) ** (1.0 / num_episodes)
    return sarsa(env, num_episodes, alpha, gamma, epsilon_start, discretizer, bins, 
                epsilon_decay, epsilon_min)

if __name__ == "__main__":
    # Blackjack
    env = gym.make('Blackjack-v1', sab=True, render_mode=None)
    Q, rewards = sarsa(env, num_episodes=10000, alpha=0.1, gamma=1.0, epsilon=0.1)
    print(f"Blackjack SARSA average reward (last 1000 episodes): {np.mean(rewards[-1000:])}")

    # CartPole (discretized)
    env = gym.make('CartPole-v1', render_mode=None)
    n_bins = 8
    bins = [np.linspace(-4.8, 4.8, n_bins),
            np.linspace(-5, 5, n_bins),
            np.linspace(-0.418, 0.418, n_bins),
            np.linspace(-5, 5, n_bins)]
    Q, rewards = sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1,
                       discretizer=discretize_state, bins=bins)
    print(f"CartPole SARSA average reward (last 1000 episodes): {np.mean(rewards[-1000:])}") 