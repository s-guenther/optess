#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import hybridsetupfactory, singlesetupfactory
from optimize_ess import OptimizeHybridESS, OptimizeSingleESS

setup_hess = hybridsetupfactory('alt.ideal', '05')
optim_hess = OptimizeHybridESS(*setup_hess)
optim_hess.results.pplot()

setup_sess = singlesetupfactory('alt.med', '1')
optim_sess = OptimizeSingleESS(*setup_sess)
optim_sess.results.pplot()

a = 'asdf'

