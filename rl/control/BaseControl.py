from abc import ABC
import numpy as np
import rl.common.Constant as CNSTNT


class BaseControl(ABC):
    def __init__(self, env, discount_factor):
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.n
        self.discount_factor = discount_factor
        self.PI = np.random.randint(0, self.action_space, size=self.state_space)
        self.Q = np.zeros((self.action_space, self.action_space))
        self.env = env

    def run(self, number_of_run):
        raise NotImplementedError('run should be implemented')

    def save_model(self):
        np.savetxt(f'../result/{type(self).__name__}_Q.txt', self.Q)
        np.savetxt(f'../result/{type(self).__name__}_PI.txt', self.PI)

    def load_model(self):
        self.PI = np.loadtxt(f'../result/{type(self).__name__}_PI.txt')
        self.Q = np.loadtxt(f'../result/{type(self).__name__}_Q.txt')

    def evaluation(self, max_step=20):
        self.load_model()
        state = self.env.reset()
        is_terminal = False
        time = 0
        while not is_terminal:
            action = np.argmax(self.Q[state, :])
            state, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
            time += 1
            if time >= max_step:
                break
        self.env.save(name=type(self).__name__)
