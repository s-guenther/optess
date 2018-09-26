#!/usr/bin/env python3

import optess as oe
import numpy as np

siggen = oe.factories.DataFactory.distorted_sin
randgen = oe.factories.DataFactory.rand

npoints = 43200
seed = npoints*2
time = 720

base = siggen(npoints=npoints, seed=seed, time=time, mu=10,
              freq=(1/168, 1/24, 1),
              amp=(0.2, 1, 0.1),
              jittervariance=(0.1/168, 0.2/24, 0.8),
              ampvariance=(0.1, 0.5, 0.4))
noise = randgen(npoints=npoints, seed=seed, time=time, mu=0,
                freq=(12,), amp=(0.1,))
sig = base + noise
base.pplot()
sig.pplot()

fft = oe.PSD(sig)
fft.pplot()

dummybreakpoint = True
