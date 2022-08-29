import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import time


def makeFig():
    plt.scatter(xList, yList)  # I think you meant this


plt.ion()  # enable interactivity
fig = plt.figure()  # make a figure

xList = list()
yList = list()

for i in np.arange(50):
    y = np.random.random()
    xList.append(i)
    yList.append(y)
    drawnow(makeFig)
    # makeFig()
    # plt.draw()
    # plt.pause(0.001)
    time.sleep(0.1)
