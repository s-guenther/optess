#!/usr/bin/env python3

import optess as oe

sig = oe.factories.DataFactory.freq(360, mu=2.0, seed=31415)

sdia = oe.SingleDia(sig)
sdia.find_pareto_front()

dummybreakpoint = True
