import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import time
from typing import List

def make_fig() -> None:
    """
    Create a scatter plot of the current data.
    
    This function is called by drawnow to update the plot
    with the current x and y data points.
    """
    plt.scatter(x_list, y_list)

def main() -> None:
    """
    Demonstrate real-time plotting using drawnow.
    
    This function creates a real-time scatter plot that updates
    as new data points are generated and added to the lists.
    """
    # Enable interactive mode for real-time plotting
    plt.ion()
    fig = plt.figure()

    # Initialize data lists
    x_list: List[int] = []
    y_list: List[float] = []

    # Generate and plot data points
    for i in np.arange(50):
        y = np.random.random()
        x_list.append(i)
        y_list.append(y)
        
        # Update the plot in real-time
        drawnow(make_fig)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
