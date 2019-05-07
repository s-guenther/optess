#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

import optess as oe

signal = oe.DataFactory.rand(200, mu=1, freq=(6, 10),
                             amp=(1, 3), time=100, seed=200)
signal.pplot()
objective = oe.Target('energy', 190)
power = 1.2
storage = oe.Storage(power, 0.95, 300)
opt = oe.OptimizeSingleESS(signal, storage, objective)
opt.solve_pyomo_model()
energy = opt.results.energycapacity
print('Single: power = {}, energy = {}'.format(power, energy))

base = oe.Storage(0.5*power, 0.95, 1e50)
peak = oe.Storage(0.5*power, 0.95, 1e50)
opthyb = oe.OptimizeHybridESS(signal, base, peak, objective)
opthyb.solve_pyomo_model()
baseenergy = opthyb.results.baseenergycapacity
peakenergy = opthyb.results.peakenergycapacity
print('Hybrid: baseenergy = {}, peakenergy = {}'.format(baseenergy,
                                                        peakenergy))

virtualbreakpoint = True
