from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import layers
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
            state = self.decode_state(self.env.reset())
            is_terminal = False
            while not is_terminal:
                action = policy.epsilon_greedy(self.exploration_rate, self.action_space, self.get_action_value(state))
                next_state, reward, is_terminal, info = self.env.step(CNSTNT.ACTIONS_VALUES[action])
                next_state = self.decode_state(next_state)
                self.add_transition(state, action, reward, next_state, is_terminal)
                self.optimize_model()
                state = next_state
                #print(f'{state},{next_state},{action}')
            total_length, total_reward, _ = self.env.history()
            runs[i] = (total_length, total_reward)
            print(f'{i}:{total_reward}')
        self.save_model()
        return runs

    def decode_state(self, state):
        # x, y = self.env.get_position(state)
        # return np.array([x, y]).reshape(1, 2)
        return self.env.get_last_frame()

    def get_action_value(self, state):
        return self._model.predict(np.array(state).reshape((1, self.env.height, self.env.width, 1)))[0]

    def add_transition(self, state, action, reward, next_state, done):
        self._memory.append(Transition(state, action, reward, next_state, done))

    def build_model(self):
        # model = Sequential()
        # model.add(Dense(32, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        # model.add(Dense(self.action_space, activation='linear', kernel_initializer='he_uniform'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        model = Sequential()
        model.add(layers.Conv2D(4, (5, 5), activation='relu', input_shape=(self.env.height, self.env.width, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (5, 5), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.env.action_space.n, activation='softmax'))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        return model

    def optimize_model(self, ):
        if len(self._memory) < self._batch_size:
            return
        batch = random.sample(self._memory, self._batch_size)
        update_input = []
        update_target = []
        for transition in batch:
            target = transition.reward
            if not transition.done:
                target = transition.reward + self.discount_factor * np.amax(self.get_action_value(transition.next_state))
            target_f = self._model.predict(np.array(transition.state).reshape((1, self.env.height, self.env.width, 1)))
            target_f[0][transition.action] = target
            update_input.append(transition.state)
            update_target.append(target_f[0])
        input_ds = np.array(update_input)
        target_ds = np.array(update_target)
        self._model.fit(input_ds.reshape((*input_ds.shape, 1)), target_ds, epochs=1, verbose=0)

    def save_model(self):
        self._model.save(f'../result/{type(self).__name__}_Weights.h5')

    def load_model(self):
        self._model = load_model(f'../result/{type(self).__name__}_Weights.h5')

    def evaluation(self, max_step=50):
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
