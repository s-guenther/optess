#!/usr/bin/env python3
"""
Optimization with pyomo library.
"""

import pyomo.environ as pe
import itertools


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


def make_model():
    """Builds model for a specific case (3x Tetris)"""
    vals = [1, 0.5] * 3 + [-1, -0.5] * 3
    n = len(vals)

    psinglecap = 1
    esinglecap = max(energy(vals))

    cut = 0.75
    pbasecap = cut * psinglecap
    ppeakcap = (1 - cut) * psinglecap
    mm = pe.ConcreteModel()

    mm.ind = pe.Set(initialize=range(n))

    mm.pbase = pe.Var(mm.ind, bounds=(-pbasecap, pbasecap))
    mm.ppeak = pe.Var(mm.ind, bounds=(-ppeakcap, ppeakcap))

    mm.ebasecap = pe.Var(bounds=(0, esinglecap))
    mm.epeakcap = pe.Var(bounds=(0, esinglecap))

    mm.obj = pe.Objective(expr=mm.epeakcap)

    mm.enrgycapequal = pe.Constraint(expr=(mm.ebasecap + mm.epeakcap ==
                                           esinglecap))

    def pwrequal_rule(model, ii):
        return model.pbase[ii] + model.ppeak[ii] == vals[ii]

    mm.pwrequal = pe.Constraint(mm.ind, rule=pwrequal_rule)

    def enrgy_below_base_rule(model, ii):
        enrgy = list(energy(model.pbase[:]))
        return enrgy[ii] <= model.ebasecap

    def enrgy_below_peak_rule(model, ii):
        enrgy = list(energy(model.ppeak[:]))
        return enrgy[ii] <= model.epeakcap

    def enrgy_base_above_zero_rule(model, ii):
        enrgy = list(energy(model.pbase[:]))
        return 0 <= enrgy[ii]

    def enrgy_peak_above_zero_rule(model, ii):
        enrgy = list(energy(model.ppeak[:]))
        return 0 <= enrgy[ii]

    mm.enrgybelowbase = pe.Constraint(mm.ind, rule=enrgy_below_base_rule)
    mm.enrgybelowpeak = pe.Constraint(mm.ind, rule=enrgy_below_peak_rule)
    mm.enrgybaseabovezero = pe.Constraint(mm.ind,
                                          rule=enrgy_base_above_zero_rule)
    mm.enrgypeakabovezero = pe.Constraint(mm.ind,
                                          rule=enrgy_peak_above_zero_rule)

    return mm


def main():
    model = make_model()
    # solver = pe.SolverFactory('glpk')
    # solver.solve(model)
    # model.pprint()
    return model

mymodel = main()
