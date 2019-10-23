import numpy as np
import rl.common.Constant as CNSTNT
from rl.common import policy, Tiling
from rl.control.BaseControl import BaseControl


class EpisodicSemiGradientSARSA(BaseControl):
    def __init__(self, env, discount_factor, exploration_rate, step_size, tiling: Tiling):
        assert isinstance(tiling, Tiling.Tiling), 'tiling parameter is not Tiling object'

        super().__init__(env, discount_factor)
        self.exploration_rate = exploration_rate
        self.step_size = step_size
        self.weight = np.zeros((env.action_space.n, tiling.tile_size()))
        self.tiling = tiling

    def run(self, number_of_episode):
        runs = {}
        for i in range(number_of_episode):
            state = self.env.reset()
            action = policy.epsilon_greedy(self.exploration_rate, self.action_space, self.get_action_value(state))
            is_terminal = False
            while not is_terminal:
                state_next, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
                action_next = policy.epsilon_greedy(self.exploration_rate, self.action_space, self.get_action_value(state_next))
                x, y = self.env.get_position(state)
                self.weight[self.tiling.tiles(x, y)] += self.step_size * (
                        reward + self.discount_factor * self.get_action_value(state_next, action_next) - self.get_action_value(state,
                                                                                                                               action))
                state = state_next
                action = action_next
            total_length, total_reward, _ = self.env.history()
            runs[i] = (total_length, total_reward)
        super().save_model()
        return runs

    def get_action_value(self, state, action=None):
        if action is None:
            action_value = []
            for action in CNSTNT.ALL_ACTIONS:
                action_value.append(self.get_action_value(state, action))
            return action_value
        else:
            x, y = self.env.get_position(state)
            return self.weight[self.tiling.tiles(x, y)].sum()

    def save_model(self):
        np.savetxt(f'../result/{type(self).__name__}_W.txt', self.weight)
