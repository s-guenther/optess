#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import optimsetupfactory
from optimmodel import OptimModel

setup = optimsetupfactory('std.ideal', '05')
optim = OptimModel(*setup)
model = optim.model
solution = optim.results
