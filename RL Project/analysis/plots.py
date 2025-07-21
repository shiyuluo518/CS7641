import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_learning_curve(rewards, title, filename, window=100):
    plt.figure()
    rewards = np.array(rewards)
    plt.plot(rewards, label='Episode Reward', alpha=0.3)
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), moving_avg, label=f'Moving Avg ({window})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_comparison(curves, labels, title, filename):
    plt.figure()
    for rewards, label in zip(curves, labels):
        plt.plot(rewards, label=label)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_vi_pi_iterations(vi_iters, pi_iters, filename):
    plt.figure()
    plt.bar(['Value Iteration', 'Policy Iteration'], [vi_iters, pi_iters])
    plt.ylabel('Iterations to Converge')
    plt.title('VI vs. PI Convergence')
    plt.savefig(filename)
    plt.close()

def plot_wall_clock_time(vi_time, pi_time, filename):
    """Plot wall-clock time comparison between VI and PI"""
    plt.figure(figsize=(8, 6))
    algorithms = ['Value Iteration', 'Policy Iteration']
    times = [vi_time, pi_time]
    colors = ['skyblue', 'lightcoral']
    
    bars = plt.bar(algorithms, times, color=colors, alpha=0.7)
    plt.ylabel('Wall-Clock Time (seconds)')
    plt.title('VI vs. PI Wall-Clock Time Comparison')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_blackjack_policy_heatmap(Q, filename_prefix):
    """Create policy heatmaps for Blackjack with and without usable ace"""
    # Extract policy from Q-table
    policy = {}
    for (state, action), value in Q.items():
        if isinstance(state, tuple) and len(state) == 3:  # (player_sum, dealer_card, usable_ace)
            player_sum, dealer_card, usable_ace = state
            if (player_sum, dealer_card, usable_ace) not in policy:
                policy[(player_sum, dealer_card, usable_ace)] = {}
            policy[(player_sum, dealer_card, usable_ace)][action] = value
    
    # Create separate heatmaps for usable ace and no usable ace
    for usable_ace in [True, False]:
        # Create policy matrix
        policy_matrix = np.zeros((18, 10))  # player_sum 4-21, dealer_card 1-10
        
        for player_sum in range(4, 22):
            for dealer_card in range(1, 11):
                if (player_sum, dealer_card, usable_ace) in policy:
                    # Find the action with highest Q-value
                    action_values = policy[(player_sum, dealer_card, usable_ace)]
                    best_action = max(action_values.items(), key=lambda x: x[1])[0]
                    policy_matrix[player_sum-4, dealer_card-1] = best_action
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(policy_matrix, 
                   xticklabels=range(1, 11),
                   yticklabels=range(4, 22),
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Action (0=Hit, 1=Stand)'},
                   annot=True, fmt='.0f')
        plt.xlabel('Dealer Showing Card')
        plt.ylabel('Player Sum')
        ace_status = "with" if usable_ace else "without"
        plt.title(f'Blackjack Optimal Policy ({ace_status} usable ace)')
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{'usable_ace' if usable_ace else 'no_usable_ace'}.png")
        plt.close()

def plot_discretization_analysis(n_bins_list, rewards_list, times_list, filename):
    """Plot the effect of discretization on performance and computation time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot final average reward
    ax1.plot(n_bins_list, rewards_list, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Bins')
    ax1.set_ylabel('Final Average Reward (last 100 episodes)')
    ax1.set_title('Effect of Discretization on Performance')
    ax1.grid(True, alpha=0.3)
    
    # Plot computation time
    ax2.plot(n_bins_list, times_list, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Bins')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Effect of Discretization on Computation Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_exploration_comparison(constant_rewards, decaying_rewards, filename):
    """Compare constant vs decaying epsilon exploration strategies"""
    plt.figure(figsize=(10, 6))
    
    # Plot both learning curves
    episodes = range(len(constant_rewards))
    plt.plot(episodes, constant_rewards, label='Constant ε=0.1', alpha=0.7)
    plt.plot(episodes, decaying_rewards, label='Decaying ε (1.0→0.05)', alpha=0.7)
    
    # Add moving averages for better visualization
    window = 100
    if len(constant_rewards) >= window:
        const_avg = np.convolve(constant_rewards, np.ones(window)/window, mode='valid')
        decay_avg = np.convolve(decaying_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(constant_rewards)), const_avg, 
                label=f'Constant Avg ({window})', linewidth=2)
        plt.plot(np.arange(window-1, len(decaying_rewards)), decay_avg, 
                label=f'Decaying Avg ({window})', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Exploration Strategy Comparison: Constant vs Decaying Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_dqn_vs_tabular(dqn_rewards, sarsa_rewards, filename):
    """Direct comparison of DQN vs tabular SARSA performance"""
    plt.figure(figsize=(10, 6))
    
    # Plot both learning curves
    dqn_episodes = range(len(dqn_rewards))
    sarsa_episodes = range(len(sarsa_rewards))
    
    plt.plot(dqn_episodes, dqn_rewards, label='DQN', alpha=0.7, color='blue')
    plt.plot(sarsa_episodes, sarsa_rewards, label='SARSA (Tabular)', alpha=0.7, color='red')
    
    # Add moving averages
    window = 50
    if len(dqn_rewards) >= window:
        dqn_avg = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(dqn_rewards)), dqn_avg, 
                label=f'DQN Avg ({window})', linewidth=2, color='darkblue')
    
    if len(sarsa_rewards) >= window:
        sarsa_avg = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(sarsa_rewards)), sarsa_avg, 
                label=f'SARSA Avg ({window})', linewidth=2, color='darkred')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN vs Tabular SARSA Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 