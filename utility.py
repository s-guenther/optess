#!/usr/bin/env/python3
"""Some utility functions used by various modules"""

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def make_empty_axes():
    ax = plt.figure().add_subplot(1, 1, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')
    return ax


def make_two_empty_axes():
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 3])
    plt.figure()
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.set_ylabel('Power')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Time')

    return ax1, ax2


def make_three_empty_axes():
    gs = gridspec.GridSpec(3, 1, height_ratios=[40, 37, 23])
    plt.figure()
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.set_label('Input Power')
    ax2.set_ylabel('Storage Power')
    ax3.set_ylabel('Energy')
    ax3.set_xlabel('Time')

    return ax1, ax2, ax3
