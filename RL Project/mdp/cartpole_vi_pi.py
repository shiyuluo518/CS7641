import gymnasium as gym
import numpy as np
from collections import defaultdict
from mdp.utils import discretize_state

def build_discretized_space(n_bins):
    bins = [np.linspace(-4.8, 4.8, n_bins),
            np.linspace(-5, 5, n_bins),
            np.linspace(-0.418, 0.418, n_bins),
            np.linspace(-5, 5, n_bins)]
    state_space = [range(n_bins) for _ in range(4)]
    return bins, state_space

def value_iteration_cartpole(env, n_bins=8, theta=1e-4, gamma=0.99):
    bins, state_space = build_discretized_space(n_bins)
    V = defaultdict(float)
    policy = defaultdict(int)
    nA = env.action_space.n
    iteration = 0
    while True:
        delta = 0
        for s0 in state_space[0]:
            for s1 in state_space[1]:
                for s2 in state_space[2]:
                    for s3 in state_space[3]:
                        state = (s0, s1, s2, s3)
                        v = V[state]
                        q_sa = np.zeros(nA)
                        for a in range(nA):
                            env.reset()
                            env.unwrapped.state = [bins[0][s0], bins[1][s1], bins[2][s2], bins[3][s3]]
                            next_state, reward, terminated, truncated, _ = env.step(a)
                            done = terminated or truncated
                            if not done:
                                next_state_disc = discretize_state(next_state, bins)
                                q_sa[a] = reward + gamma * V[next_state_disc]
                            else:
                                q_sa[a] = reward
                        V[state] = np.max(q_sa)
                        policy[state] = np.argmax(q_sa)
                        delta = max(delta, abs(v - V[state]))
        iteration += 1
        if delta < theta:
            break
    return V, policy, iteration

def policy_iteration_cartpole(env, n_bins=8, theta=1e-4, gamma=0.99):
    bins, state_space = build_discretized_space(n_bins)
    policy = defaultdict(lambda: 0)
    V = defaultdict(float)
    pi_iterations = 0
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for s0 in state_space[0]:
                for s1 in state_space[1]:
                    for s2 in state_space[2]:
                        for s3 in state_space[3]:
                            state = (s0, s1, s2, s3)
                            v = V[state]
                            a = policy[state]
                            env.reset()
                            env.unwrapped.state = [bins[0][s0], bins[1][s1], bins[2][s2], bins[3][s3]]
                            next_state, reward, terminated, truncated, _ = env.step(a)
                            done = terminated or truncated
                            if not done:
                                next_state_disc = discretize_state(next_state, bins)
                                V[state] = reward + gamma * V[next_state_disc]
                            else:
                                V[state] = reward
                            delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
        # Policy Improvement
        policy_stable = True
        for s0 in state_space[0]:
            for s1 in state_space[1]:
                for s2 in state_space[2]:
                    for s3 in state_space[3]:
                        state = (s0, s1, s2, s3)
                        old_action = policy[state]
                        q_sa = np.zeros(env.action_space.n)
                        for a in range(env.action_space.n):
                            env.reset()
                            env.unwrapped.state = [bins[0][s0], bins[1][s1], bins[2][s2], bins[3][s3]]
                            next_state, reward, terminated, truncated, _ = env.step(a)
                            done = terminated or truncated
                            if not done:
                                next_state_disc = discretize_state(next_state, bins)
                                q_sa[a] = reward + gamma * V[next_state_disc]
                            else:
                                q_sa[a] = reward
                        best_action = np.argmax(q_sa)
                        policy[state] = best_action
                        if old_action != best_action:
                            policy_stable = False
        pi_iterations += 1
        if policy_stable:
            break
    return V, policy, pi_iterations

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode=None)
    print("Running Value Iteration on CartPole (discretized)...")
    V, policy, vi_iterations = value_iteration_cartpole(env, n_bins=6)
    print("Value Iteration complete.")
    print("Running Policy Iteration on CartPole (discretized)...")
    V2, policy2, pi_iterations = policy_iteration_cartpole(env, n_bins=6)
    print("Policy Iteration complete.") 