# GirdWorld
a multi purpose environment based on gridworld environment which was used in 'Reinforcement Learning: An Introduction' book. The environment provide a dynamic grid word compatible with openAI GYM

# Define environment

```json
{
  "width": 12,
  "height": 4,
  "save_frame": true,
  "reward": -1,
  "states": {
    "25,26,27,28,29,30,31,32,33,34": {"D": [-100,"36"]},
    "36": {"start":true,"R": [-100,"36"]},
    "37,38,39,40,41,42,43,44,45,46": {"color": "khaki","*": ["_","36"]},
    "47": {"terminal": true}
  }
}
```
```python
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
```
<p  align="center">
<img src="/doc/img/cliff.gif" />
</p>

# Q-Learning
```python
environment = MDPGrid(json_path=os.path.join(os.getcwd(), f'../env/cliff.json'))
learner = QLearning(environment, discount_factor=gamma, exploration_rate=epsilon, step_size=alpha)
learner.run(number_of_episode)
learner.evaluation()
environment.save()
```
<p  align="center">
<img src="/doc/img/cliff-q.gif" />
</p>