import time
import gymnasium as gym
import numpy as np
from mdp.sarsa import sarsa, sarsa_with_decaying_epsilon
from mdp.cartpole_vi_pi import value_iteration_cartpole, policy_iteration_cartpole
from mdp.dqn import dqn_cartpole
from mdp.utils import discretize_state
from analysis.plots import (plot_learning_curve, plot_vi_pi_iterations, plot_wall_clock_time,
                          plot_blackjack_policy_heatmap, plot_discretization_analysis,
                          plot_exploration_comparison, plot_dqn_vs_tabular)

if __name__ == "__main__":
    print("=== Blackjack SARSA ===")
    env = gym.make('Blackjack-v1', sab=True, render_mode=None)
    Q, rewards = sarsa(env, num_episodes=10000, alpha=0.1, gamma=1.0, epsilon=0.1)
    print(f"Blackjack SARSA average reward (last 1000 episodes): {np.mean(rewards[-1000:])}")
    print(f"Sample Blackjack SARSA rewards: {rewards[:20]}")
    plot_learning_curve(rewards, "Blackjack SARSA Learning Curve", "analysis/blackjack_sarsa_curve.png", window=100)
    
    # Generate Blackjack policy heatmaps
    print("Generating Blackjack policy heatmaps...")
    plot_blackjack_policy_heatmap(Q, "analysis/blackjack_policy")

    print("\n=== CartPole SARSA (discretized) ===")
    env = gym.make('CartPole-v1', render_mode=None)
    n_bins = 8
    bins = [np.linspace(-4.8, 4.8, n_bins),
            np.linspace(-5, 5, n_bins),
            np.linspace(-0.418, 0.418, n_bins),
            np.linspace(-5, 5, n_bins)]
    Q, rewards = sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1,
                       discretizer=discretize_state, bins=bins)
    print(f"CartPole SARSA average reward (last 1000 episodes): {np.mean(rewards[-1000:])}")
    print(f"Sample CartPole SARSA rewards: {rewards[:20]}")
    plot_learning_curve(rewards, "CartPole SARSA Learning Curve", "analysis/cartpole_sarsa_curve.png", window=100)

    print("\n=== CartPole Value Iteration vs Policy Iteration (Wall-Clock Time) ===")
    env = gym.make('CartPole-v1', render_mode=None)
    
    # Time Value Iteration
    start_time = time.time()
    V, policy, vi_iterations = value_iteration_cartpole(env, n_bins=6)
    vi_time = time.time() - start_time
    print(f"Value Iteration complete. Iterations: {vi_iterations}, Time: {vi_time:.2f}s")

    # Time Policy Iteration
    start_time = time.time()
    V2, policy2, pi_iterations = policy_iteration_cartpole(env, n_bins=6)
    pi_time = time.time() - start_time
    print(f"Policy Iteration complete. Iterations: {pi_iterations}, Time: {pi_time:.2f}s")
    
    # Plot both iterations and wall-clock time
    plot_vi_pi_iterations(vi_iterations, pi_iterations, "analysis/vi_pi_iterations.png")
    plot_wall_clock_time(vi_time, pi_time, "analysis/vi_pi_wall_clock_time.png")

    print("\n=== Discretization Analysis ===")
    n_bins_list = [4, 6, 8, 10, 12]
    discretization_rewards = []
    discretization_times = []
    
    for n_bins in n_bins_list:
        print(f"Testing with {n_bins} bins...")
        bins = [np.linspace(-4.8, 4.8, n_bins),
                np.linspace(-5, 5, n_bins),
                np.linspace(-0.418, 0.418, n_bins),
                np.linspace(-5, 5, n_bins)]
        
        start_time = time.time()
        Q, rewards = sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1,
                           discretizer=discretize_state, bins=bins)
        computation_time = time.time() - start_time
        
        final_reward = np.mean(rewards[-100:])  # Average of last 100 episodes
        discretization_rewards.append(final_reward)
        discretization_times.append(computation_time)
        print(f"  Final reward: {final_reward:.2f}, Time: {computation_time:.2f}s")
    
    plot_discretization_analysis(n_bins_list, discretization_rewards, discretization_times, 
                               "analysis/discretization_analysis.png")

    print("\n=== Exploration Strategy Comparison ===")
    # Constant epsilon
    Q_const, rewards_const = sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.99, epsilon=0.1,
                                   discretizer=discretize_state, bins=bins)
    
    # Decaying epsilon
    Q_decay, rewards_decay = sarsa_with_decaying_epsilon(env, num_episodes=5000, alpha=0.1, gamma=0.99,
                                                        epsilon_start=1.0, epsilon_min=0.05,
                                                        discretizer=discretize_state, bins=bins)
    
    print(f"Constant epsilon final reward: {np.mean(rewards_const[-100:]):.2f}")
    print(f"Decaying epsilon final reward: {np.mean(rewards_decay[-100:]):.2f}")
    
    plot_exploration_comparison(rewards_const, rewards_decay, "analysis/exploration_comparison.png")

    print("\n=== CartPole DQN (extra credit) ===")
    dqn_rewards = dqn_cartpole(num_episodes=500)
    print(f"DQN CartPole average reward (last 100 episodes): {np.mean(dqn_rewards[-100:])}")
    plot_learning_curve(dqn_rewards, "CartPole DQN Learning Curve", "analysis/cartpole_dqn_curve.png", window=50)
    
    # Direct comparison of DQN vs SARSA
    print("\n=== DQN vs Tabular SARSA Comparison ===")
    # Use the same number of episodes for fair comparison
    sarsa_comparison_rewards = rewards[:len(dqn_rewards)]  # Truncate to match DQN episodes
    plot_dqn_vs_tabular(dqn_rewards, sarsa_comparison_rewards, "analysis/dqn_vs_tabular_comparison.png")
    
    print("\n=== Summary ===")
    print(f"VI vs PI Wall-clock time: VI={vi_time:.2f}s, PI={pi_time:.2f}s")
    print(f"Best discretization performance: {max(discretization_rewards):.2f} with {n_bins_list[np.argmax(discretization_rewards)]} bins")
    print(f"Exploration comparison: Constant={np.mean(rewards_const[-100:]):.2f}, Decaying={np.mean(rewards_decay[-100:]):.2f}")
    print(f"DQN vs SARSA: DQN={np.mean(dqn_rewards[-100:]):.2f}, SARSA={np.mean(sarsa_comparison_rewards[-100:]):.2f}") 