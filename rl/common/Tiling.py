import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Tiling:
    def __init__(self, width: int, height: int, number_of_tilling: int, bin: float, offset: tuple):
        self._width = width
        self._height = height
        self._number_of_tiling = number_of_tilling
        self._bin = bin
        self._offset_x, self._offset_y = offset
        self._tiling = []

    def create_tiling_grid(self, tiling_index):
        grid_x = np.arange(0, self._width + self._bin + 1, self._bin) + self._offset_x + tiling_index
        grid_y = np.arange(0, self._height + self._bin + 1, self._bin) + self._offset_y + tiling_index
        grid = np.array([np.clip(grid_x, 0, self._width), np.clip(grid_y, 0, self._height)])
        return grid

    def create_tilings(self):
        self._tiling = [self.create_tiling_grid(tiling_index) for tiling_index in range(self._number_of_tiling)]

    def tile_encode(self, x, y):
        zero = np.zeros(self._width, self._height)
        return [(int(np.digitize(x, grid[0])) - 1, int(np.digitize(y, grid[1])) - 1) for grid in self._tiling]

    def visualize_tilings(self):
        """Plot each tiling as a grid."""
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyles = ['-', '--', '-.', ':']
        legend_lines = []

        fig, ax = plt.subplots(figsize=(10, 10))
        for i, grid in enumerate(self._tiling):
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
