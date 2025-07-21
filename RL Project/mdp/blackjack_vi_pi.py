import gymnasium as gym
import numpy as np
from collections import defaultdict

def value_iteration(env, theta=1e-6, gamma=1.0):
    V = defaultdict(float)
    policy = defaultdict(int)
    nA = env.action_space.n
    while True:
        delta = 0
        for state in env.observation_space:
            v = V[state]
            q_sa = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in env.P[state][a]:
                    q_sa[a] += prob * (reward + gamma * V[next_state] * (not done))
            V[state] = np.max(q_sa)
            policy[state] = np.argmax(q_sa)
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V, policy

def policy_evaluation(env, policy, gamma=1.0, theta=1e-6):
    V = defaultdict(float)
    while True:
        delta = 0
        for state in env.observation_space:
            v = V[state]
            a = policy[state]
            v_new = 0
            for prob, next_state, reward, done in env.P[state][a]:
                v_new += prob * (reward + gamma * V[next_state] * (not done))
            V[state] = v_new
            delta = max(delta, abs(v - v_new))
        if delta < theta:
            break
    return V

def policy_iteration(env, gamma=1.0, theta=1e-6):
    policy = defaultdict(lambda: env.action_space.sample())
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        policy_stable = True
        for state in env.observation_space:
            old_action = policy[state]
            q_sa = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                for prob, next_state, reward, done in env.P[state][a]:
                    q_sa[a] += prob * (reward + gamma * V[next_state] * (not done))
            best_action = np.argmax(q_sa)
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
        if policy_stable:
            break
    return V, policy

if __name__ == "__main__":
    env = gym.make('Blackjack-v1', sab=True, render_mode=None)
    # Value Iteration
    # Note: Gymnasium's Blackjack-v1 does not expose env.P or env.observation_space as iterable, so this is a placeholder.
    print("Blackjack Value Iteration and Policy Iteration should be run using model-based MDPs. For Gym's Blackjack, use Monte Carlo or SARSA.") 