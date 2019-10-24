import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from rl.common.Tiling import Tiling
from rl.control.EpisodicSemiGradientSARSA import EpisodicSemiGradientSARSA
from rl.control.QLearning import QLearning
from rl.environment.mdp.MDPGrid import MDPGrid

alpha = 0.5 / 8
gamma = 1
epsilon = 0.1
number_of_episode = 100
number_of_run = 30
environment = MDPGrid(json_path=os.path.join(os.getcwd(), f'../env/env_10.json'))
tiling = Tiling(environment.width, environment.height, number_of_tilling=4, bin=4, offset=(-3, -3))
environment.render()
step_per_episode = np.zeros(number_of_episode)

for run in tqdm(range(number_of_run)):
    learner = EpisodicSemiGradientSARSA(environment, discount_factor=gamma, exploration_rate=epsilon, step_size=alpha, tiling=tiling)
    runs_history = learner.run(number_of_episode)
    for episode, item in runs_history.items():
        step_per_episode[episode] += item[0]

step_per_episode /= number_of_run

plt.plot(step_per_episode, label="Semi Gradient SARSA")
plt.xlabel('Episode')
plt.ylabel('Step per episode')
plt.legend()
plt.show()
# learner.evaluation()
# environment.save()
