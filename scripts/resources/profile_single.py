#!/usr/bin/env python3
"""
Creates a random signal with n points points and initializes the hybriddia
object, i.e. calculates the single storage optimisation
"""

import timeit
import sys

import optess as oe


def main(npoints, nprocesses):
    # Define and create signal
    start = timeit.default_timer()

    mu = 100
    freq = [int(npoints*f) for f in [0.005, 0.02, 0.05, 0.2]]
    ampl = [1, 2, 4, 8]
    time = 3600
    seed = npoints*10

    signal = oe.DataFactory.rand(npoints, mu, freq, ampl, time, seed)

    # Read some Signal infor
    pmin = signal.amv
    pmax = max(signal.vals)
    peakcut = (pmax - pmin)*0.4 + pmin
    storagepower = pmax - peakcut

    end = timeit.default_timer() - start
    print('Preprocessing... {} seconds'.format(end))

    # Define, create objective and storage
    objective = oe.Objective('power', peakcut)
    storage = oe.Storage(storagepower, 0.95, 1e6)

    name = 'single{}'.format(npoints)

    # Start calculation
    start = timeit.default_timer()
    dia = oe.HybridDia(signal, storage, objective,
                       name=name, nprocesses=nprocesses)
    end = timeit.default_timer() - start
    print('Single Calculation ... {} seconds'.format(end))

    # Save
    start = timeit.default_timer()
    dia.save()
    end = timeit.default_timer() - start
    print('Saving... {} seconds'.format(end))


if __name__ == '__main__':
    NPOINTS = int(sys.argv[1])
    NPROCESSES = 1
    main(NPOINTS, NPROCESSES)
