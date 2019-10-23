import os

from rl.common.Tiling import Tiling
from rl.control.EpisodicSemiGradientSARSA import EpisodicSemiGradientSARSA
from rl.control.QLearning import QLearning
from rl.environment.mdp.MDPGrid import MDPGrid

alpha = 0.5 / 8
gamma = 0.9
epsilon = 0.1
number_of_episode = 500

environment = MDPGrid(json_path=os.path.join(os.getcwd(), f'../env/env_10.json'))
# environment.render()
#
tiling = Tiling(environment.width, environment.height, number_of_tilling=4, bin=4, offset=(-3, -3))
# print(tiling.tiles(11, 5))
learner = EpisodicSemiGradientSARSA(environment, discount_factor=gamma, exploration_rate=epsilon, step_size=alpha, tiling=tiling)
learner.run(number_of_episode)
learner.evaluation()
# environment.save()
