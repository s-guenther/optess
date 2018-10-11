#!/usr/bin/env python3

import optess as oe
from optess.hybridanalysis import HybridAnalysis


hyb = oe.HybridDia.load('sig_5000_noise_50_amp_100')
hybana = HybridAnalysis(hyb)

dummybreakpoint = True
