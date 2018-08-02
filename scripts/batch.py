#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import DataFactory, ObjectiveFactory, HybridSetupFactory,\
                      SingleSetupFactory
from optimize_ess import OptimizeHybridESS, OptimizeSingleESS
from storage import Storage
from objective import Objective
from hybriddia import HybridDia
import timeit

# storage = Storage(2, 0.95, 0.01)
# storage = Storage(2.0, 0.95, 1e-2)
# objective = ObjectiveFactory.std03()

# start = timeit.default_timer()
# setup_sess = SingleSetupFactory.std()
# optim_sess = OptimizeSingleESS(*setup_sess)
# optim_sess.pplot()
# end = timeit.default_timer() - start
# print('SESS Calculation went for {} seconds'.format(end))
#
# start = timeit.default_timer()
# setup_hess = HybridSetupFactory.std()
# optim_hess = OptimizeHybridESS(*setup_hess)
# optim_hess.pplot()
# end = timeit.default_timer() - start
# print('HESS Calculation went for {} seconds'.format(end))

peakcut = 3
objective = Objective('power', peakcut)
for npoints in [44, 66, 88, 110, 220, 440]:
    name = 'std_{}'.format(npoints)
    signal = DataFactory.std(npoints)
    storagepower = (max(signal.vals) - peakcut)
    storage = Storage(storagepower, 0.95, 1000)
    start = timeit.default_timer()
    dia = HybridDia(signal, storage, objective, name=name)
    dia.calculate_curves()
    dia.calculate_area((7, 7))
    end = timeit.default_timer() - start
    print('Hybrid Dia Calculation npoints = {} went for {} seconds'.format(
        npoints, end))
    dia.pplot()
    dia.save()

virtualbreakpoint = True
