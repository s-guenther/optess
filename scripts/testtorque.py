#!/usr/bin/env python3

import optess as oe

sig = oe.DataFactory.rand(npoints=32, seed=123)
obj = oe.Objective('power', 12)
stor = oe.Storage(max(sig.vals) - 12, 0.95, 1e6)
hyb = oe.HybridDia(sig, stor, obj)
hyb.compute_torque()

dummybreakpoint = True
