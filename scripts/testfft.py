#!/usr/bin/env python3
"""Performs and fft of a signal"""

import numpy as np
from matplotlib import pyplot as plt

import optess as oe

npoints = 9700
siglength = 97
x = np.linspace(siglength/npoints, siglength, npoints)
y = np.sin(2*np.pi*x) + \
    np.sin(5*2*np.pi*x + np.pi/2) + \
    0.5*np.sin(20*2*np.pi*x + np.pi/2)

sigs = []
ffts = []
for seed in [1234, 1337]:
    sig = oe.DataFactory.rand(npoints=2**12, ampl=[1, 1, 1], time=1,
                              freq=[50, 300, 1500], seed=seed, mu=0)
    sigs.append(sig)
    ffts.append(oe.FFT(sig))

virtualbreakpoint = True
for sig, fft in zip(sigs, ffts):
    sig.pplot()
    fft.pplot(plot=plt.loglog)

virtualbreakpoint = True


