import os

from rl.control.QLearning import QLearning
from rl.environment.mdp.MDPGrid import MDPGrid

alpha = 0.1
gamma = 1
epsilon = 0.1
number_of_episode = 1000

environment = MDPGrid(json_path=os.path.join(os.getcwd(), f'../env/cliff.json'))
learner = QLearning(environment, discount_factor=gamma, exploration_rate=epsilon, step_size=alpha)
learner.run(number_of_episode)
learner.evaluation()
environment.save()
