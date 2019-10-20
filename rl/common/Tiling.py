import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


class Tiling:
    def __init__(self, width: int, height: int, number_of_tilling: int, bin: float, offset: tuple):
        self._width = width
        self._height = height
        self._number_of_tiling = number_of_tilling
        self._bin = bin
        self._offset_x, self._offset_y = offset

    def create_tiling_grid(self, offset):
        grid = [np.linspace(0, self._width, self._bin + 1)[1:] + self._offset_x + offset,
                np.linspace(0, self._height, self._bin + 1)[1:] + self._offset_y + offset
                ]
        return grid

    def create_tilings(self):
        return [self.create_tiling_grid(offset=(index)) for index in range(self._number_of_tiling)]

    def visualize_tilings(self, tilings):
        """Plot each tiling as a grid."""
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyles = ['-', '-', '-']
        legend_lines = []

        fig, ax = plt.subplots(figsize=(10, 10))
        for i, grid in enumerate(tilings):
            for x in grid[0]:
                l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
            for y in grid[1]:
                l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
            legend_lines.append(l)
        ax.grid('off')
        ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
        ax.set_title("Tilings")
        plt.show()
        return ax  # return Axis object to draw on later, if needed
