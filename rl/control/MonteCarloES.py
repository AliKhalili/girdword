"""Monte Carlo ES (Exploring Starts), for estimating optimal PI"""
import numpy as np
from rl.control.BaseControl import BaseControl
from tqdm import tqdm


class MonteCarloES(BaseControl):
    def __init__(self, env, discount_factor):
        super().__init__(env, discount_factor)

    def run(self, number_of_run):
        for i in tqdm(range(number_of_run)):
            state = self.env.reset()
            action = np.random.choice(self.action_space)

            is_terminal = False
            trajectory = np.empty((0, 5), dtype=int)
            time_step = 1
            epsilon_rate = 1 - i / number_of_run
            while not is_terminal:
                state_next, reward, is_terminal, info = self.env.step(action)
                trajectory = np.append(trajectory, np.array([[time_step, state, action, reward, state_next]], dtype=int), axis=0)
                state = state_next
                action = self.PI[state]
                if np.random.binomial(1, epsilon_rate) == 1:
                    action = np.random.choice(self.action_space)
                time_step += 1
            G = 0

            for time, state, action, reward, state_next in trajectory[::-1]:
                G = self.discount_factor * G + reward
                if np.where(trajectory[time:, 1] == state)[0].size == 0:
                    self.Q[state, action] = self.Q[state, action] + (1 / time) * (G - self.Q[state, action])
                    self.PI[state] = np.argmax(self.Q[state, :])
        super().save_model()
