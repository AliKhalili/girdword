import numpy as np
from tqdm import tqdm
from rl.prediction.PredictionBase import PredictionBase
import rl.common.Constant as CNSTNT


class TabularTDZero(PredictionBase):
    def __init__(self, env, discount_factor, step_size):
        super().__init__(env, discount_factor, step_size)
        self.trajectory = {}

    def run(self, number_of_run):
        for episode in range(number_of_run):
            is_terminal = False
            state = self.env.reset()
            while not is_terminal:
                action = np.random.choice(self.action_space, 1, p=self.PI[state])[0]
                state_next, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
                self.V[state] = self.V[state] + self.step_size * (reward + self.discount_factor * self.V[state_next] - self.V[state])
                state = state_next
            self.trajectory[episode] = self.V.copy()
        super()._save()
        return self.trajectory
