import numpy as np


def epsilon_greedy(exploration_rate: int, action_space: int, Q: []):
    if np.random.binomial(1, exploration_rate) == 1:
        return np.random.choice(action_space)
    return np.random.choice(np.where(Q == np.max(Q))[0])
