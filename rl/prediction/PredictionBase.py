import numpy as np


class PredictionBase:
    def __init__(self, env, discount_factor, step_size):
        self.action_space, self.state_space = env.action_space.n, env.observation_space.n
        self.V = np.zeros(self.state_space)
        self.PI = np.ones((self.state_space, self.action_space)) / self.action_space  # uniform policy
        self.env = env
        self.discount_factor = discount_factor
        self.step_size = step_size

    def _save(self):
        np.savetxt(f'result/{type(self).__name__}_V.txt', self.V, fmt='%0.2f')

    def run(self, number_of_run):
        pass
