#!/usr/bin/env python3
"""
Loads a predefined factory setup, feeds optimization class and prints some
outputs.
"""

from factories import hybridsetupfactory, singlesetupfactory,\
                      datafactory, objectivefactory
# from optimize_ess import OptimizeHybridESS, OptimizeSingleESS
from storage import Storage
from hybriddia import HybridDia
import timeit

signal = datafactory('alt', 66)
# storage = Storage(2, 0.95, 0.01)
storage = Storage(2.1, 0.98, 0)
objective = objectivefactory('std0-3')

# start = timeit.default_timer()
# setup_hess = hybridsetupfactory('alt.low', '05')
# optim_hess = OptimizeHybridESS(*setup_hess)
# optim_hess.pplot()
# end = timeit.default_timer() - start
# print('HESS Calculation went for {} seconds'.format(end))
#
# start = timeit.default_timer()
# setup_sess = singlesetupfactory('alt.low', '2')
# optim_sess = OptimizeSingleESS(*setup_sess)
# optim_sess.pplot()
# end = timeit.default_timer() - start
# print('SESS Calculation went for {} seconds'.format(end))

start = timeit.default_timer()
dia = HybridDia(signal, storage, objective)
dia.calculate_curves()
dia.calculate_area()
end = timeit.default_timer() - start
print('Hybrid Dia Calculation went for {} seconds'.format(end))
dia.pplot()

a = 'asdf'
