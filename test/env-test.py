import os
from rl.environment.mdp.MDPGrid import MDPGrid

environment = MDPGrid(json_path=os.path.join(os.getcwd(), f'../env/env_10.json'))

environment.render()