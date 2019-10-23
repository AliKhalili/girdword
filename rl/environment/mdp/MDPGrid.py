import json
import numpy as np

from rl.environment.mdp.State import State, states_parser
from rl.environment.mdp.MDP import MDP
from rl.environment.mdp.Frame import Frame
import rl.common.Constant as CNSTNT
from rl.environment.mdp.Trajectory import Trajectory


class MDPGrid(MDP, Frame, Trajectory):
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as json_file:
            configuration = json.load(json_file)
        self.width = configuration.get('width', 0)
        self.height = configuration.get('height', 0)
        save_frame = configuration.get('save_frame', False)
        self.save_trajectory = configuration.get('save_frame', False)

        super(MDPGrid, self).__init__(number_of_state=self.height * self.width, save_frame=save_frame, save_trajectory=self.save_trajectory)
        reward = configuration.get("reward", -1)

        self._states, self.start, self.grid_move = states_parser(configuration.get("states", None), reward, self.height, self.width)
        super().state(self.start)
        super().add_frame(super().time(), self._get_frame())

    def _get_frame(self):
        frame = np.array([CNSTNT.COLORS[state.get_color()] for index, state in self._states.items()]).reshape(
            (self.height, self.width))
        frame[self.grid_move.get_position(super().state())] = CNSTNT.COLORS[CNSTNT.ACTION_COLORS["start"]]
        return frame

    def step(self, action_name):
        next_state, reward, is_terminal, time_step = super().step(action_name)
        super().add_frame(time_step, self._get_frame())
        if self.save_trajectory:
            super().add_trajectory(time_step, reward, super().state(), action_name, next_state)
        return next_state, reward, is_terminal, time_step

    def get_position(self, state):
        return self.grid_move.get_position(state)

    def render(self, time_step=None):
        if time_step is None:
            super().render(self.time())
        else:
            super().render(time_step)

    def reset(self):
        MDP.reset(self)
        Frame.reset(self)
        Trajectory.reset(self)
        state = self.state(self.start)
        super().add_frame(self.time(), self._get_frame())
        return state
