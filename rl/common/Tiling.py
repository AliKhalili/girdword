import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#TODO:change to tiles
class Tiling:
    def __init__(self, width: int, height: int, number_of_tilling: int, bin: float, offset: tuple):
        self._width = width
        self._height = height
        self._number_of_tiling = number_of_tilling
        self._bin = bin
        self._offset_x, self._offset_y = offset
        self._tiling = {}
        self._tiling_states = {}
        self._one_hot = {}
        self._tiles = {}
        self._create_tilings()

    def _create_tiling_grid(self, tiling_index):
        assert self._offset_x + tiling_index <= 0, "x offset can not be greater than zero"
        assert self._offset_y + tiling_index <= 0, "y offset can not be greater than zero"
        grid_x = np.arange(0, self._width + self._bin + 1, self._bin) + self._offset_x + tiling_index
        grid_y = np.arange(0, self._height + self._bin + 1, self._bin) + self._offset_y + tiling_index
        grid = np.array([np.unique(np.clip(grid_x, 0, self._width)), np.unique(np.clip(grid_y, 0, self._height))])
        return grid

    def _create_tilings(self):
        self._tiling = {tiling_index: self._create_tiling_grid(tiling_index) for tiling_index in range(self._number_of_tiling)}
        self._tiling_states = {
            tiling_index: np.arange((grid[1].shape[0] - 1) * (grid[0].shape[0] - 1)).reshape((grid[1].shape[0] - 1, grid[0].shape[0] - 1))
            for
            tiling_index, grid in
            self._tiling.items()}

        max_len = self._tile_size()
        for x in range(self._width):
            for y in range(self._height):
                one_hot = np.zeros((self._number_of_tiling, max_len), dtype=int)
                states = self._tile_encode(x, y)
                one_hot[np.arange(self._number_of_tiling), states] = 1
                self._one_hot[(x, y)] = one_hot.flatten()
                self._tiles[(x, y)] = states

    def _tile_encode(self, x, y):
        return np.array(
            [self._tiling_states[tiling_index][int(np.digitize(y, grid[1])) - 1, int(np.digitize(x, grid[0])) - 1] for
             tiling_index, grid
             in
             self._tiling.items()])

    def _tile_size(self):
        return max([items.size for key, items in self._tiling_states.items()])

    def size(self):
        return self._number_of_tiling * self._tile_size()

    def ont_hot(self, x, y):
        return self._one_hot[(x, y)]

    # def tiles(self, x, y):
    #     return self._tiles[(x, y)]

    def visualize_tilings(self):
        """Plot each tiling as a grid."""
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyles = ['-', '--', '-.', ':']
        legend_lines = []

        fig, ax = plt.subplots(figsize=(10, 10))
        for i, grid in self._tiling.items():
            for x in grid[0]:
                l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
            for y in grid[1]:
                l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)],
                               linewidth=1)
            legend_lines.append(l)
        ax.grid('off')
        ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white',
                  framealpha=0.9)
        ax.set_title("Tilings")
        plt.show()
        return ax
