#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

import optess as oe

signal = oe.DataFactory.rand(200, mu=1, freq=(6, 10),
                          ampl=(1, 3), time=100, seed=200)
signal.pplot()
objective = oe.Objective('energy', 170)
power = 2.1
storage = oe.Storage(power, 0.95, 1e50)
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
