import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import rl.common.Constant as CNSTNT


class Frame:
    def __init__(self, **kwargs):
        super(Frame, self).__init__()
        self._frame = {}
        self.save_frame = kwargs["save_frame"]
        self._cmap = colors.ListedColormap(CNSTNT.COLORS.keys())
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._norm = colors.BoundaryNorm(bounds, self._cmap.N)

    def reset(self):
        self._frame = {}

    def add_frame(self, time_step: int, frame):
        if self.save_frame:
            self._frame[time_step] = frame

    def render(self, time_step):
        if self.save_frame:
            fig, ax = plt.subplots()
            frame = self._frame[time_step]
            ax.imshow(frame, cmap=self._cmap, norm=self._norm)
            plt.axis('off')
            plt.show()

    def save(self, name: str):
        fig, ax = plt.subplots()
        im = ax.imshow(self._frame[0], cmap=self._cmap, norm=self._norm)
        update = lambda i: im.set_array(self._frame[i])
        anim = FuncAnimation(fig, update, frames=np.arange(1, len(self._frame)), interval=200)
        anim.save(f'../result/save_{type(self).__name__ if name is None else name}.gif', writer='pillow', fps=5)

        # ax.pcolormesh(np.transpose(frame), cmap=self._cmap, norm=self._norm, linewidths=.05, edgecolors='black')
        # ax.set_xticks(np.arange(frame.shape[1]))
        # ax.set_yticks(np.arange(frame.shape[0]))
        # ax.set_xticklabels(np.arange(frame.shape[1]))
        # ax.set_yticklabels(np.arange(frame.shape[0]))
        # plt.setp(ax.get_xticklabels(), ha="right",va="center")
        #
        # for i in range(frame.shape[0]):
        #     for j in range(frame.shape[1]):
        #         text = ax.text(j, i, frame[i, j],
        #                        horizontalalignment='center',
        #                        verticalalignment='center', color="black")
