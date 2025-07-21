# RL Project: MDPs with Dynamic Programming and Reinforcement Learning

## Overview
This project explores Markov Decision Processes (MDPs) using Value Iteration, Policy Iteration, SARSA, and Deep Q-Networks (DQN) on two environments:
- **Blackjack-v1** (Discrete, Stochastic)
- **CartPole-v1** (Continuous, Deterministic, discretized)

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run scripts from the project root.

## Scripts
- `main.py`: Entry point to run experiments and analysis.
- All algorithm implementations are in the `mdp/` directory.

## Algorithms Implemented
- Value Iteration (VI)
- Policy Iteration (PI)
- SARSA (tabular)
- DQN (PyTorch, for CartPole, optional)

## How to Run
```bash
python main.py
```

You can modify `main.py` to select which environment and algorithm to run, or run all experiments and generate plots for analysis.

## Output
- Plots and statistics will be saved in the `analysis/` directory.
- Console output will summarize convergence and performance.

## Extra Credit
- DQN and Rainbow DQN variants for CartPole are included for extra credit.

--- 