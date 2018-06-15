#!/usr/bin/env/python3
"""Some utility functions used by various modules"""

from matplotlib import pyplot as plt


def make_empty_axes():
    ax = plt.figure().add_subplot(1, 1, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')
    return ax
