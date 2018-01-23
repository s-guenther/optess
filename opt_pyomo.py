#!/usr/bin/env python3
"""
Optimization with pyomo library.
"""

from datetime import datetime
import pyomo.environ as pe
import itertools
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total


def energy(power, dtime=None):
    """energy for each timestep"""
    if dtime is None:
        dtime = itertools.repeat(1)
    return cumsum(pwr * dtme for pwr, dtme in zip(power, dtime))


def load_data(filename):
    raw = scio.loadmat(filename)
    vals = [float(val) for val in raw['vals'].flatten()]
    dtime = float(raw['dtime'].flatten())
    dtime_iter = itertools.repeat(dtime)
    return vals, dtime_iter


def make_model(vals=None, dtime=None):
    """Builds model for a specific case (3x Tetris)"""

    if vals is None:
        vals = [1, 0.5] * 3 + [-1, -0.5] * 3
        # xx = np.linspace(2*np.pi/100, 2*np.pi, 100)
        # vals = np.sin(xx) + np.sin(3*xx)
    if dtime is None:
        dtime = itertools.repeat(1)
    n = len(vals)

    psinglecap = np.max(vals)
    esinglecap = max(energy(vals, dtime))

    cut = 0.3
    pbasecap = cut * psinglecap
    ppeakcap = (1 - cut) * psinglecap

    mm = pe.ConcreteModel(name='Hybridisation')

    mm.ind = pe.Set(initialize=range(n))

    mm.pbase = pe.Var(mm.ind, bounds=(-pbasecap, pbasecap))
    mm.ppeak = pe.Var(mm.ind, bounds=(-ppeakcap, ppeakcap))

    mm.ebasecap = pe.Var(bounds=(0, esinglecap))
    mm.epeakcap = pe.Var(bounds=(0, esinglecap))

    mm.obj = pe.Objective(expr=mm.epeakcap)

    mm.enrgycapequal = pe.Constraint(expr=(mm.ebasecap + mm.epeakcap ==
                                           esinglecap))
    print('   {:%H:%M:%S}: Energycap Equal Constraint Done'.format(datetime.now()))

    def pwrequal_rule(model, ii):
        return model.pbase[ii] + model.ppeak[ii] == vals[ii]

    mm.pwrequal = pe.Constraint(mm.ind, rule=pwrequal_rule)
    print('   {:%H:%M:%S}: Power Equal Rule Done'.format(datetime.now()))

    def enrgy_below_base_rule(model, ii):
        enrgy = list(energy(model.pbase[:], dtime))
        return enrgy[ii] <= model.ebasecap

    def enrgy_below_peak_rule(model, ii):
        enrgy = list(energy(model.ppeak[:], dtime))
        return enrgy[ii] <= model.epeakcap

    def enrgy_base_above_zero_rule(model, ii):
        enrgy = list(energy(model.pbase[:], dtime))
        return -0.02 <= enrgy[ii]

    def enrgy_peak_above_zero_rule(model, ii):
        enrgy = list(energy(model.ppeak[:], dtime))
        return 0 <= enrgy[ii]

    print('   {:%H:%M:%S}: Rule Fcn Def Done'.format(datetime.now()))

    mm.enrgybelowbase = pe.Constraint(mm.ind, rule=enrgy_below_base_rule)
    print('   {:%H:%M:%S}: Energy Below Base Done'.format(datetime.now()))
    mm.enrgybelowpeak = pe.Constraint(mm.ind, rule=enrgy_below_peak_rule)
    print('   {:%H:%M:%S}: Energy Below Peak Done'.format(datetime.now()))
    mm.enrgybaseabovezero = pe.Constraint(mm.ind,
                                          rule=enrgy_base_above_zero_rule)
    print('   {:%H:%M:%S}: Energy Base Above Zero Done'.format(datetime.now()))
    mm.enrgypeakabovezero = pe.Constraint(mm.ind,
                                          rule=enrgy_peak_above_zero_rule)
    print('   {:%H:%M:%S}: Energy Peak Above Zero Done'.format(datetime.now()))

    return mm, vals, n


def main(filename='rand03.mat'):
    print('{:%H:%M:%S}: Loading Files...'.format(datetime.now()))
    vals, dtime = load_data(filename)
    print('{:%H:%M:%S}: Making Model...'.format(datetime.now()))
    model, vals, n = make_model(vals, dtime)
    print('{:%H:%M:%S}: Setting up Solver...'.format(datetime.now()))
    solver = pe.SolverFactory('gurobi')
    print('{:%H:%M:%S}: Solving Model...'.format(datetime.now()))
    solver.solve(model)
    # model.pprint()
    print('{:%H:%M:%S}: Starting Post Processing...'.format(datetime.now()))
    xx = np.linspace(2*np.pi/n, 2*np.pi, n)
    plt.step(xx, vals)
    plt.step(xx, model.ppeak.get_values().values())
    plt.step(xx, model.pbase.get_values().values())
    ebasecap = model.ebasecap.get_values().values()
    epeakcap = model.epeakcap.get_values().values()
    print('\n   Base Energy Capacity = {}'.format(ebasecap))
    print('   Peak Energy Capacity = {}'.format(epeakcap))
    return model


model = main()
