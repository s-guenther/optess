#!/usr/bin/env python3

import optess as oe
from optess import signalanalysis

freqfact = oe.factories.DataFactory.freq

for ii in range(10):
    sig = freqfact()
    sigpsd = signalanalysis.FFT(sig)
    sig.pplot()
    sigpsd.pplot()

dummybreakpoint = True
