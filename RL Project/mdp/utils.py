import numpy as np

def discretize_state(state, bins):
    """
    Discretize a continuous state into a tuple of bin indices.
    Args:
        state: np.array, the continuous state
        bins: list of np.array, each array contains the bin edges for one dimension
    Returns:
        tuple of discretized indices
    """
    return tuple(int(np.digitize(s, b) - 1) for s, b in zip(state, bins))

def epsilon_greedy(Q, state, nA, epsilon):
    """
    Epsilon-greedy action selection.
    Args:
        Q: dict or np.array, Q-table
        state: current state
        nA: number of actions
        epsilon: exploration rate
    Returns:
        action: int
    """
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    else:
        if isinstance(Q, dict):
            return np.argmax([Q.get((state, a), 0) for a in range(nA)])
        else:
            return np.argmax(Q[state]) 