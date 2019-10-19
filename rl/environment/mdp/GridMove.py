import numpy as np


class GridMove:
    def __init__(self, grid):
        self._Grid = grid
        self._height = grid.shape[0]
        self._width = grid.shape[1]

    def _in_bound(self, x, y):
        if x < 0 or x >= self._height:
            return False
        if y < 0 or y >= self._width:
            return False
        return True

    def get_next(self, action_name, index):
        move = lambda index, x, y: self._Grid[x, y] if self._in_bound(x, y) else index
        switcher = {
            "UP": lambda index, x, y: move(index, x - 1, y),
            "RIGHT": lambda index, x, y: move(index, x, y + 1),
            "DOWN": lambda index, x, y: move(index, x + 1, y),
            "LEFT": lambda index, x, y: move(index, x, y - 1),

            "U": lambda index, x, y: move(index, x - 1, y),
            "R": lambda index, x, y: move(index, x, y + 1),
            "D": lambda index, x, y: move(index, x + 1, y),
            "L": lambda index, x, y: move(index, x, y - 1),
        }
        move_func = switcher.get(action_name)
        x, y = self.get_position(index)
        return move_func(index, x, y)

    def get_position(self, index):
        result_index = np.where(self._Grid == index)
        return result_index[0][0], result_index[1][0]
