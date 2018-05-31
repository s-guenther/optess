#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import optimsetupfactory
from optimmodel import OptimModel
from matplotlib import pyplot as plt

ax = plt.figure().add_subplot(1, 1, 1)

setup = optimsetupfactory('std.ideal', '05')
setup.signal.pplot(color='g', ax=ax)
optim = OptimModel(*setup)
model = optim.model
solution = optim.results

solution.base.pplot(ax=ax)

a = 'asdf'
