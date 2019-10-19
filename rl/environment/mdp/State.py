import numpy as np
import rl.common.Constant as CNSTNT

from rl.common.Utils import intTryParse
from rl.environment.mdp.GridMove import GridMove


class State:
    def __init__(self, index: int, is_terminal: bool, color: str):
        self._index = index
        self._is_terminal = is_terminal
        self._action = {}
        self._color = color

    def index(self):
        return self._index

    def is_terminal(self):
        return self._is_terminal

    def add_action(self, action_name: str, reward: int, next_state: int):
        self._action[action_name] = (next_state, reward)

    def get_actions(self):
        return self._action

    def step(self, action_name):
        return self._action[action_name]

    def get_color(self):
        return self._color

    def __str__(self):
        a = ",".join([f'({action}:{reward},{next_state})' for action, (next_state, reward) in self._action.items()])
        return f'{self._index} : {a}{"terminal" if self._is_terminal else ""}'


all_actions = ["U", "R", "D", "L"]


def states_parser(states_dict: dict, reward: int, height: int, width: int):
    if not isinstance(states_dict, dict):
        raise TypeError("input must be a dictionary")

    grid = np.arange(height * width).reshape((height, width))
    grid_move = GridMove(grid)

    states = {}
    start_state = None

    for group_of_states_key, group_of_states_value in states_dict.items():
        all_states_in_group = group_of_states_key.split(',')
        is_terminal = group_of_states_value.get("terminal", False)
        is_start = group_of_states_value.get("start", False)

        color = group_of_states_value.get("color", "lightgray")
        current_actions = []
        if "*" in group_of_states_value.keys():
            current_actions = all_actions.copy()
        else:
            for default_action in all_actions:
                if default_action in group_of_states_value.keys():
                    current_actions.append(default_action)

        for selected_state_in_group in all_states_in_group:
            new_state = State(int(selected_state_in_group), is_terminal, color if not is_terminal else CNSTNT.ACTION_COLORS["terminal"])
            if is_start:
                start_state = new_state.index()
            if not is_terminal:
                for action in all_actions:
                    if action not in current_actions:
                        new_state.add_action(action, reward, grid_move.get_next(action, new_state.index()))
                    else:
                        overwrite_reward, next_actions = group_of_states_value.get("*", group_of_states_value.get(action, (None, None)))
                        overwrite_reward, *_ = (reward, 0) if overwrite_reward == '_' else intTryParse(overwrite_reward)
                        possible_state_index, result_parse = intTryParse(next_actions)
                        if result_parse:
                            new_state.add_action(action, overwrite_reward, possible_state_index)
                        else:
                            next_state = new_state.index()
                            for n_action in next_actions:
                                n_action = action if n_action == "_" else n_action
                                next_state = grid_move.get_next(n_action, next_state)
                            new_state.add_action(action, overwrite_reward, next_state)
            states[new_state.index()] = new_state

    np_it = np.nditer(grid, flags=['multi_index'])
    while not np_it.finished:
        index = np_it.iterindex
        if index not in states:
            new_state = State(int(index), False, CNSTNT.ACTION_COLORS["other"])
            for action in all_actions:
                new_state.add_action(action, reward, grid_move.get_next(action, new_state.index()))
            states[new_state.index()] = new_state
        #print(states[index])
        np_it.iternext()
    return {key: states[key] for key in sorted(states)}, start_state, grid_move
