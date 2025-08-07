from typing import List

import matplotlib
import matplotlib.pyplot as plt
from drawnow import drawnow
from matplotlib.figure import Figure


class RealTimePlot:
    """
    Real-time plotting utility for visualizing data during training.
    
    This class provides functionality to create and update real-time plots
    for monitoring training progress or other time-series data.

    Attributes
    ----------
    fig : Figure
        The matplotlib figure object.
    ax : plt.Axes
        The matplotlib axes object.
    values : List[float]
        List of values to plot.
    background : matplotlib.backend_bases.FigureCanvasBase
        Background for efficient redrawing.
    line : matplotlib.lines.Line2D
        The line object for plotting.
    """
    def __init__(self) -> None:
        """
        Initialize the RealTimePlot with a matplotlib figure and axes.
        """
        self.fig, self.ax = plt.subplots(1, 1)
        self.values: List[float] = []
        
        # Initialize the plot
        self.fig.canvas.draw()
        plt.show(block=False)
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.line = self.ax.plot(self.values, range(len(self.values)))[0]

    def update(self, value: float) -> None:
        """
        Update the plot with a new value.

        Parameters
        ----------
        value : float
            The new value to add to the plot.
        """
        self.values.append(value)
        self.line.set_data(self.values, range(len(self.values)))
        self.fig.canvas.restore_region(self.background)
        
        # Redraw just the points
        self.ax.draw_artist(self.line)
        
        # Fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

    def reset(self) -> None:
        """
        Reset the plot by clearing all values.
        """
        self.values: List[float] = []
