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
        self.weight = np.zeros((self.action_space, tiling.size()))
        self.tiling = tiling

    def run(self, number_of_episode):
        runs = {}
        for i in range(number_of_episode):
            one_hot = self.decode_state(self.env.reset())
            action = policy.epsilon_greedy(self.exploration_rate, self.action_space, self.get_action_value(one_hot))
            is_terminal = False
            while not is_terminal:
                state_next, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
                one_hot_next = self.decode_state(state_next)
                action_next, target = 0, 0
                if not is_terminal:
                    action_values = self.get_action_value(one_hot_next)
                    action_next = policy.epsilon_greedy(self.exploration_rate, self.action_space, action_values)
                    target = action_values[action_next]

                estimation = self.get_action_value(one_hot, action)
                self.weight[action] += self.step_size * (reward + self.discount_factor * target - estimation) * one_hot

                one_hot = one_hot_next
                action = action_next
            total_length, total_reward, _ = self.env.history()
            #print(f'{i}:{total_length}')
            runs[i] = (total_length, total_reward)
        self.save_model()
        return runs

    def get_action_value(self, one_hot, action=None):
        if action is None:
            action_value = []
            for action in CNSTNT.ALL_ACTIONS:
                action_value.append(self.weight[CNSTNT.ACTIONS[action]].dot(one_hot))
            return action_value
        else:
            return self.weight[action].dot(one_hot)

    def decode_state(self, state):
        x, y = self.env.get_position(state)
        return self.tiling.ont_hot(x, y)

    def save_model(self):
        np.savetxt(f'../result/{type(self).__name__}_W.txt', self.weight)

    def load_model(self):
        self.PI = np.loadtxt(f'../result/{type(self).__name__}_W.txt')

    def evaluation(self, max_step=30):
        self.load_model()
        state = self.env.reset()
        is_terminal = False
        time = 0
        while not is_terminal:
            one_hot = self.decode_state(state)
            action = np.argmax(self.get_action_value(one_hot))
            state, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
            time += 1
            if time >= max_step:
                break
        self.env.save(name=type(self).__name__)
