import gym
from rl.environment.mdp.State import State
import rl.common.Constant as Constant


class MDP:
    def __init__(self, **kwargs):
        super(MDP, self).__init__(**kwargs)
        self._states = {}
        self._cursor = 0
        self._time_step = 0
        self.action_space = gym.spaces.Discrete(len(Constant.ACTIONS))
        self.observation_space = gym.spaces.Discrete(kwargs["number_of_state"])

    def add_state(self, state: State):
        self._states[state.index()] = state

    def time(self):
        return self._time_step

    def step(self, action_name):
        next_state, reward = self._states[self.state()].step(action_name)
        self.state(next_state)
        self._time_step += 1
        return next_state, reward, self._states[next_state].is_terminal(), self._time_step

    def state(self, state=None):
        if state is None:
            return self._cursor
        self._cursor = state
        return self._cursor

    def is_terminal(self, state):
        return self._states[state].is_terminal()

    def reset(self):
        self._time_step = 0
