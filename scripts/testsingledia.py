#!/usr/bin/env python3

import optess as oe

sig = oe.factories.DataFactory.freq(3600, mu=1.0, seed=31415, time=100)

sdia = oe.SingleDia(sig, rel_obj_fronts=[0.2, 0.5, 0.8])
sdia.compute_parallel()
sdia.pplot()
sdia.save()

dummybreakpoint = True
