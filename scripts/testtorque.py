#!/usr/bin/env python3

import optess as oe

pointslist = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
pointslist += [1500, 3000, 6000, 12000, 24000, 48000, 96000, 192000]
pointslist += [13000, 13001, 30000, 30001, 55000, 55001, 115000, 115001,
               175000, 175001]

for npoints in pointslist:
    # Define signal
    mu = 100
    freq = [int(npoints*f) for f in [0.005, 0.02, 0.05, 0.2]]
    ampl = [1, 2, 4, 8]
    time = 3600
    seed = npoints*10
    signal = oe.DataFactory.rand(npoints, mu, freq, ampl, time, seed)

    # Read some Signal info
    pmin = signal.amv
    pmax = max(signal.vals)
    peakcut = (pmax - pmin)*0.4 + pmin
    storagepower = pmax - peakcut

    objective = oe.Target('power', peakcut)
    storage = oe.Storage(storagepower, 0.95, 1e7)

    name = 'signal_{}'.format(npoints)

    hyb = oe.HybridDia(signal, storage, objective, name=name)
    print('Submit {}'.format(npoints))
    hyb.compute_torque(wt=1.5, mem=1.2, returninfo=False)

dummybreakpoint = True
