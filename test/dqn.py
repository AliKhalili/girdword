import os

from rl.control.DQN import DQN
from rl.control.QLearning import QLearning
from rl.environment.mdp.MDPGrid import MDPGrid

alpha = 0.1
gamma = 0.9
learning_rate = 0.001
epsilon = 0.1
number_of_episode = 10

environment = MDPGrid(json_path=os.path.join(os.getcwd(), f'../env/env_10.json'))
learner = DQN(environment, discount_factor=gamma, exploration_rate=epsilon, step_size=alpha, learning_rate=learning_rate)
learner.run(number_of_episode)
learner.evaluation()
environment.save()
