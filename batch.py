#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import DataFactory, ObjectiveFactory, HybridSetupFactory,\
                      SingleSetupFactory
from optimize_ess import OptimizeHybridESS, OptimizeSingleESS
from storage import Storage
from hybriddia import HybridDia
import timeit

signal = DataFactory.std(44)
# storage = Storage(2, 0.95, 0.01)
storage = Storage(2.0, 0.95, 1e-2)
objective = ObjectiveFactory.std03()

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

start = timeit.default_timer()
dia = HybridDia(signal, storage, objective)
dia.calculate_curves()
dia.calculate_area()
end = timeit.default_timer() - start
print('Hybrid Dia Calculation went for {} seconds'.format(end))
dia.pplot()
dia.save('testfile.hyb')

virtualbreakpoint = True
