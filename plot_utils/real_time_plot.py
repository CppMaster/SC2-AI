from typing import List

import matplotlib
import matplotlib.pyplot as plt
from drawnow import drawnow
from matplotlib.figure import Figure


class RealTimePlot:

    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)
        self.values: List[float] = []
        # plt.draw()
        self.fig.canvas.draw()
        plt.show(block=False)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.line = self.ax.plot(self.values, range(len(self.values)))[0]

    def update(self, value: float):
        self.values.append(value)
        self.line.set_data(self.values, range(len(self.values)))
        self.fig.canvas.restore_region(self.background)
        # redraw just the points
        self.ax.draw_artist(self.line)
        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

    def reset(self):
        self.values: List[float] = []
