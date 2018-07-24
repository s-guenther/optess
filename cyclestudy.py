#!/usr/bin/env python3
"""
Generates some random signals and solves them.
"""

import itertools
from factories import DataFactory
from objective import Objective
from optimize_ess import OptimizeHybridESS, OptimizeSingleESS
from storage import Storage
from hybriddia import HybridDia
from matplotlib import pyplot as plt
import timeit

hybdias = list()
seeds = [314159]
noises = [0, 0.5, 1, 1.5, 2, 2.5, 3, 5]

npoints = 600
mu = 40
freqs = [3, 10, 100]
ampl = [0.3, 2]

peakcut = 40.5
objective = Objective('power', peakcut)

for seed, noise in itertools.product(seeds, noises):
    print('-----  Calculating seed={}, noise={}   -----'.format(seed, noise))
    sig = DataFactory.rand(npoints, mu, freqs, ampl + [noise], seed=seed)
    storagepower = max(sig.vals) - peakcut
    storage = Storage(storagepower, 0.95, 1e-3)
    hyb = HybridDia(sig, storage, objective)
    hyb.calculate_curves()
    hyb.calculate_area((11, 11))
    hybdias.append(hyb)

for hyb, (seed, noise) in zip(hybdias, itertools.product(seeds, noises)):
    hyb.pplot()
    plt.savefig('hyb_seed_{}_noise_{:.2f}.png'.format(seed, noise),
                bbox_inches='tight')
    plt.savefig('hyb_seed_{}_noise_{:.2f}.eps'.format(seed, noise),
                bbox_inches='tight')

dummybreakpoint = None
