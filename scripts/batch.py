#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

import timeit

import optess as oe

peakcut = 3
objective = oe.Objective('power', peakcut)
for npoints in [44, 66, 88, 110, 220, 440]:
    name = 'std_{}'.format(npoints)
    signal = oe.DataFactory.std(npoints)
    storagepower = (max(signal.vals) - peakcut)
    storage = oe.Storage(storagepower, 0.95, 1000)
    start = timeit.default_timer()
    dia = oe.HybridDia(signal, storage, objective, name=name)
    dia.calculate_curves()
    dia.calculate_area((7, 7))
    end = timeit.default_timer() - start
    print('Hybrid Dia Calculation npoints = {} went for {} seconds'.format(
        npoints, end))
    dia.pplot()
    dia.save()

virtualbreakpoint = True
