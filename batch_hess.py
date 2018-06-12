#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import hybridsetupfactory, singlesetupfactory
from optimhybridees import OptimHybridEES
from optimsingleees import OptimSingleEES

setup_hess = hybridsetupfactory('std.ideal', '05')
optim_hess = OptimHybridEES(*setup_hess)
model_hess = optim_hess.model
solution_hess = optim_hess.results
solution_hess.pplot()

setup_sess = singlesetupfactory('std.ideal', '05')
optim_sess = OptimSingleEES(*setup_sess)
model_sess = optim_sess.model
solution_sess = optim_sess.results
solution_sess.pplot()

a = 'asdf'

