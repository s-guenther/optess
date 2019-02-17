#!/usr/bin/env python3

import optess as oe


sig = oe.DataFactory.distorted_sin(npoints=100, mu=0, seed=30349)
obj = oe.Objective('exact', 0)
stor = oe.Storage(max(abs(sig.vals)), 0.95, 500)

hyb = oe.HybridDia(sig, stor, obj)
hyb.compute_parallel(cuts=[0.25, 0.5, 0.75], curves=2, workers=4)
hyb.pplot()

dummybreakpoint = True
