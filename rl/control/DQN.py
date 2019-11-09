from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from collections import namedtuple
import random
import numpy as np

from rl.common import policy
from rl.control.BaseControl import BaseControl
import rl.common.Constant as CNSTNT

Transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(BaseControl):
    def __init__(self, env, discount_factor, exploration_rate, step_size, learning_rate):
        super().__init__(env, discount_factor)
        self.exploration_rate = exploration_rate
        self.step_size = step_size
        self.learning_rate = learning_rate
        self._model = self.build_model()
        self._batch_size = 128
        self._memory = deque(maxlen=2000)

    def run(self, number_of_episode):
        runs = {}
        for i in range(number_of_episode):
            state = self.env.reset()
            is_terminal = False
            while not is_terminal:
                action = policy.epsilon_greedy(self.exploration_rate, self.action_space, self.get_action_value(state))
                next_state, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])

                self.add_transition(state, action, reward, next_state, is_terminal)
                state = next_state
                self.optimize_model()

            total_length, total_reward, _ = self.env.history()
            runs[i] = (total_length, total_reward)
        self.save_model()
        return runs

    def decode_state(self, state):
        x, y = self.env.get_position(state)
        return np.array([x, y]).reshape(1, 2)

    def get_action_value(self, state):
        return self._model.predict(self.decode_state(state))[0]

    def add_transition(self, state, action, reward, next_state, done):
        self._memory.append(Transition(state, action, reward, next_state, done))

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_space, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def optimize_model(self, ):
        if len(self._memory) < self._batch_size:
            return
        batch = random.sample(self._memory, self._batch_size)
        for transition in batch:
            target = transition.reward
            if not transition.done:
                target = transition.reward + self.discount_factor * np.max(self.get_action_value(transition.next_state))
            target_f = self._model.predict(self.decode_state(transition.state))
            target_f[0][transition.action] = target
            self._model.fit(self.decode_state(transition.state), target_f, epochs=1, verbose=0)

    def save_model(self):
        self._model.save(f'../result/{type(self).__name__}_Weights.h5')

    def load_model(self):
        self._model = load_model(f'../result/{type(self).__name__}_Weights.h5')

    def evaluation(self, max_step=20):
        self.load_model()
        state = self.env.reset()
        is_terminal = False
        time = 0
        while not is_terminal:
            action = np.argmax(self._model.predict(self.decode_state(state))[0])
            state, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
            time += 1
            if time >= max_step:
                break
        self.env.save(name=type(self).__name__)
