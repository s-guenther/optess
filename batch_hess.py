#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import optimsetupfactory, datafactory
from optimmodel import OptimModel
from matplotlib import pyplot as plt

alt = datafactory('alt', 128)
altmod = -(alt > 4)*(4 + alt)

setup = optimsetupfactory('std.ideal', '05')
optim = OptimModel(*setup)
model = optim.model
solution = optim.results
solution.pplot()

a = 'asdf'

