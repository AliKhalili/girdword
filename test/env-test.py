import os
from rl.environment.mdp.MDPGrid import MDPGrid

environment = MDPGrid(json_path=os.path.join(os.getcwd(), '../env/cliff.json'))
environment.step('U')
environment.step('U')
environment.step('R')
environment.step('R')
environment.step('R')
environment.step('R')
environment.step('R')
environment.step('R')
environment.step('D')
environment.step('D')
environment.save()
