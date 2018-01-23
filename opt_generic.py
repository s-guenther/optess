#!/usr/bin/env python3
"""
generic single storage optimization, where a problem or load profile is
defined by an upper and lower bound of a discretized power profile. The
single storage power profile is optimized. Aim is to minimize the needed
energy capacity of the single storage.
"""

from datetime import datetime
import pyomo.environ as pe
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt


def std_values(highcut=3, lowcut=0):
    """Provide example data when called as standalone script"""
    x = range(0, 17)
    y = [3, 5, 3, 2, 3, 4, 3, 0, -1, 0, 3, 5, 3, -2, 3, 2, 2]
    xx = np.linspace(16/128, 16, 128)
    spl = interp.PchipInterpolator(x, y)
    yy = spl(xx)
    lower = list(-(yy - lowcut))
    upper = list(-(yy - highcut))
    return lower, upper, xx


def make_model(lower, upper, efficiency=0.999, disch=1e-4, maxpwr=2, dtime=1):
    """Builds a pyomo model, taking lower and upper as input and minimizes
    storage dimension"""
    if not all(lwr <= upr for lwr, upr in zip(lower, upper)):
        raise ValueError('lower > upper at at least one point')

    mm = pe.ConcreteModel(name='Single Storage Optimization')

    mm.ind = pe.Set(initialize=range(len(lower)), ordered=True)

    # define pwr bounds
    lwrbnds = [max(-maxpwr, lwr) for lwr in lower]
    uprbnds = [min(maxpwr, upr) for upr in upper]

    if not all(lwrbnd <= uprbnd for lwrbnd, uprbnd in zip(lwrbnds, uprbnds)):
        raise ValueError('maxpwr not sufficiently large for given data set')

    # create variable pwr with bounds
    def bnds(model, ii):
        return lwrbnds[ii], uprbnds[ii]
    mm.pwr = pe.Var(mm.ind, bounds=bnds)

    # create variable pwrplus with bounds
    def bndsplus(model, ii):
        return 0, None
    mm.pwrplus = pe.Var(mm.ind, bounds=bndsplus)

    # create variable pwrminus with bounds
    def bndsminus(model, ii):
        return None, 0
    mm.pwrminus = pe.Var(mm.ind, bounds=bndsminus)

    # create variable pwrideal
    mm.pwrideal = pe.Var(mm.ind)

    # create variable energy with bounds as constraint (bounded by other
    # variable 'energycap' - energy capacity)
    mm.enrgy = pe.Var(mm.ind, bounds=(0, None))
    mm.maxenrgy = pe.Var(bounds=(0, None))

    def energylowermax(model, ii):
        return model.enrgy[ii] <= model.maxenrgy
    mm.enrgylwrmax = pe.Constraint(mm.ind, rule=energylowermax)

    # create variable for initial condition
    mm.startenrgy = pe.Var(bounds=(0, None))
    mm.startlowermax = pe.Constraint(expr=(mm.startenrgy <= mm.maxenrgy))

    # cyclic constraint startenergy = endenergy TODO soft sufficient?
    mm.cyclic = pe.Constraint(expr=(mm.startenrgy == mm.enrgy[mm.ind.last()]))
    # mm.cyclic1 = pe.Constraint(expr=(mm.startenrgy - mm.enrgy[mm.ind.last()]
    #                                  <= mm.maxenrgy/100))
    # mm.cyclic2 = pe.Constraint(expr=(mm.startenrgy - mm.enrgy[mm.ind.last()]
    #                                  >= -mm.maxenrgy/100))

    # constraint split power in positive and negative part
    def powerequal(model, ii):
        return model.pwr[ii] == model.pwrplus[ii] + model.pwrminus[ii]
    mm.pwrequal = pe.Constraint(mm.ind, rule=powerequal)

    # constraint self discharge and efficiency losses
    def losses(model, ii):
        eff_losses = (model.pwrplus[ii]*efficiency +
                      model.pwrminus[ii]/efficiency)

        if ii is 0:
            disch_losses = -model.startenrgy*disch
        else:
            disch_losses = -model.enrgy[ii-1]*disch
        return model.pwrideal[ii] == eff_losses + disch_losses
    mm.pwrafterlosses = pe.Constraint(mm.ind, rule=losses)

    # constraint integrate energy - connect power and energy
    def integrate_power(model, ii):
        if ii is 0:
            return mm.enrgy[ii] == mm.startenrgy + mm.pwrideal[ii]*dtime
        else:
            return mm.enrgy[ii] == mm.enrgy[ii-1] + mm.pwrideal[ii]*dtime
    mm.intpwr = pe.Constraint(mm.ind, rule=integrate_power)

    # objective
    objective_expr = mm.maxenrgy
    multiplier = 1
    penalty_expr = sum(mm.pwrplus[ii] - mm.pwrminus[ii] for ii in mm.ind)
    mm.obj = pe.Objective(expr=(objective_expr + multiplier*penalty_expr))

    return mm


def main():
    print('{:%H:%M:%S}: Making Model...'.format(datetime.now()))
    lwr, upr, xx = std_values()
    model = make_model(lwr, upr)

    print('{:%H:%M:%S}: Setting up Solver...'.format(datetime.now()))
    solver = pe.SolverFactory('gurobi')

    print('{:%H:%M:%S}: Solving Model...'.format(datetime.now()))
    solver.solve(model)

    print('{:%H:%M:%S}: Starting Post Processing...'.format(datetime.now()))
    plt.figure()
    plt.step(xx, upr, color='r', linestyle='--')
    plt.step(xx, lwr, color='g', linestyle='--')
    plt.step(xx, model.pwr.get_values().values(), color='b')
    plt.step(xx, model.pwrideal.get_values().values(), color='b',
             linestyle='--')
    plt.grid()
    plt.figure()
    plt.step(xx, model.enrgy.get_values().values(), color='b')
    plt.grid()
    # ebasecap = MODEL.ebasecap.get_values().values()
    # epeakcap = MODEL.epeakcap.get_values().values()
    # print('\n   Base Energy Capacity = {}'.format(ebasecap))
    # print('   Peak Energy Capacity = {}'.format(epeakcap))


if __name__ == '__main__':
    main()