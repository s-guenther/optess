#!/usr/bin/env python3
"""
Creates a random signal with 1000 points and calculates the hybridisation
curve, but not the area
"""

import timeit
import sys

import optess as oe


def main(npoints, nprocesses):
    mu = 100
    freq = [int(npoints*f) for f in [0.005, 0.02, 0.05, 0.2]]
    ampl = [1, 2, 4, 8]
    time = 3600
    seed = npoints

    signal = oe.DataFactory.rand(npoints, mu, freq, ampl, time, seed)
    pmin = signal.amv
    pmax = max(signal.vals)
    peakcut = (pmax - pmin)*0.3 + pmin
    storagepower = pmax - peakcut

    objective = oe.Objective('power', peakcut)
    storage = oe.Storage(storagepower, 0.95, 1e5)
    name = 'all{}'.format(npoints)

    start = timeit.default_timer()
    dia = oe.HybridDia(signal, storage, objective,
                       name=name, nprocesses=nprocesses)
    end = timeit.default_timer() - start
    print('Single Calc went for {} seconds'.format(end))

    start = timeit.default_timer()
    dia.calculate_curves()
    end = timeit.default_timer() - start
    print('Curve Calculation went for {} seconds'.format(end))

    start = timeit.default_timer()
    dia.calculate_area((11, 11))
    end = timeit.default_timer() - start
    print('Area Calculation went for {} seconds'.format(end))
    print('Calculated points: {}'.format(len(dia.area.keys())))

    dia.save()


if __name__ == '__main__':
    NPOINTS = int(sys.argv[1])
    PROCESSORS = int(sys.argv[2])
    main(NPOINTS, PROCESSORS)
