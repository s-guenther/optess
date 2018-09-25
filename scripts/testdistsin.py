#!/usr/bin/env python3

import optess as oe
import numpy as np

siggen = oe.factories.DataFactory.distorted_sin


# sig = siggen(npoints=8192,
#              freq=(3, 21, 100),
#              ampl=np.array((1, 5, 0.1))*np.sqrt(2),
#              time=10,
#              seed=1234)
sig = siggen(npoints=10000, freq=(4,), ampl=(np.sqrt(2),), time=100,
             seed=12345, variance=0.2, jitter=0.5, mu=0)
sig.pplot()

fft = oe.PSD(sig)
fft.pplot()

dummybreakpoint = True
