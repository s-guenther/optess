#!/usr/bin/env python3
"""
Generates some random signals and solves them.
"""

import itertools
from matplotlib import pyplot as plt

import optess as oe

hybdias = list()
seeds = [1234]
noises = [0, 1]

npoints = 256
mu = 40
freqs = [3, 10, 50]
ampl = [0.3, 2]

peakcut = 41
objective = oe.Objective('power', peakcut)

for seed, noise in itertools.product(seeds, noises):
    print('-----  Calculating seed={}, noise={}   -----'.format(seed, noise))
    sig = oe.DataFactory.rand(npoints, mu, freqs, ampl + [noise], seed=seed)
    storagepower = max(sig.vals) - peakcut
    storage = oe.Storage(storagepower, 0.95, 1e16)
    hyb = oe.HybridDia(sig, storage, objective)
    hyb.calculate_curves([0.4, 0.6, 0.8])
    hybdias.append(hyb)

for hyb, (seed, noise) in zip(hybdias, itertools.product(seeds, noises)):
    hyb.pplot()
    plt.savefig('hyb_seed_{}_noise_{:.2f}.png'.format(seed, noise),
                bbox_inches='tight')
    plt.savefig('hyb_seed_{}_noise_{:.2f}.eps'.format(seed, noise),
                bbox_inches='tight')

dummybreakpoint = None
