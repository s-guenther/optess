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
import time

from opt_generic import std_values, alt_values


def make_model(lower, upper, efficiency=0.99, disch=1e-2, maxpwr_base=0.6,
               maxpwr_peak=1.4, dtime=1):
    """Builds a pyomo model, taking lower and upper as input and minimizes
    storage dimension, for 2 storage system"""
    if not all(lwr <= upr for lwr, upr in zip(lower, upper)):
        raise ValueError('lower > upper at at least one point')

    mm = pe.ConcreteModel(name='Hybrid Storage Optimization')

    mm.ind = pe.Set(initialize=range(len(lower)), ordered=True)

    # define pwr bounds
    lwrbnds = [max(-(maxpwr_base + maxpwr_peak), lwr) for lwr in lower]
    uprbnds = [min(maxpwr_base + maxpwr_peak, upr) for upr in upper]

    if not all(lwrbnd <= uprbnd for lwrbnd, uprbnd in zip(lwrbnds, uprbnds)):
        raise ValueError('maxpwr not sufficiently large for given data set')

    # create variables pwrbase and pwrpeak with bounds
    mm.pwrbase = pe.Var(mm.ind, bounds=(-maxpwr_base, maxpwr_base))
    mm.pwrpeak = pe.Var(mm.ind, bounds=(-maxpwr_peak, maxpwr_peak))

    # create variable pwrplusbase and pwrpluspeak with bounds
    mm.pwrplusbase = pe.Var(mm.ind, bounds=(0, maxpwr_base))
    mm.pwrpluspeak = pe.Var(mm.ind, bounds=(0, maxpwr_peak))

    # create variable pwrminusbase and pwrminuspeak with bounds
    mm.pwrminusbase = pe.Var(mm.ind, bounds=(-maxpwr_base, 0))
    mm.pwrminuspeak = pe.Var(mm.ind, bounds=(-maxpwr_peak, 0))

    # create variable pwridealbase and pwridealpeak
    mm.pwridealbase = pe.Var(mm.ind)
    mm.pwridealpeak = pe.Var(mm.ind)

    # create variable pwrinter for inter-storage power flow w/o losses
    mm.pwrinter = pe.Var(mm.ind)  # set this to zero to prevent reloading
    # mm.pwrinter = pe.Var(mm.ind, bounds=(0, 0))

    # constraint that pwrinter + pwrinout stays within limits
    def interconstraintbaserule(model, ii):
        expr = model.pwrinter[ii] + model.pwrplusbase[ii] + \
               model.pwrminusbase[ii]
        return -maxpwr_base/efficiency, expr, maxpwr_base*efficiency
    mm.interbase = pe.Constraint(mm.ind, rule=interconstraintbaserule)

    def interconstraintpeakrule(model, ii):
        expr = -model.pwrinter[ii] + model.pwrpluspeak[ii] + \
               model.pwrminuspeak[ii]
        return -maxpwr_peak/efficiency, expr, maxpwr_peak*efficiency
    mm.interpeak = pe.Constraint(mm.ind, rule=interconstraintpeakrule)

    # create variable energy base and peak with bounds as constraint (bounded
    # by other variable 'energycap' - energy capacity)
    mm.enrgybase = pe.Var(mm.ind, bounds=(0, None))
    mm.maxenrgybase = pe.Var(bounds=(0, None))
    mm.enrgypeak = pe.Var(mm.ind, bounds=(0, None))
    mm.maxenrgypeak = pe.Var(bounds=(0, None))

    def energybaselowermax(model, ii):
        return model.enrgybase[ii] <= model.maxenrgybase

    def energypeaklowermax(model, ii):
        return model.enrgypeak[ii] <= model.maxenrgypeak

    mm.enrgybaselwrmax = pe.Constraint(mm.ind, rule=energybaselowermax)
    mm.enrgypeaklwrmax = pe.Constraint(mm.ind, rule=energypeaklowermax)

    # create variable for initial condition (base and peak)
    mm.startenrgybase = pe.Var(bounds=(0, None))
    mm.startenrgypeak = pe.Var(bounds=(0, None))
    mm.startlowermaxbase = pe.Constraint(expr=(mm.startenrgybase <=
                                               mm.maxenrgybase))
    mm.startlowermaxpeak = pe.Constraint(expr=(mm.startenrgypeak <=
                                               mm.maxenrgypeak))

    # constraint added powers within lower/upper bounds
    def powerswithinbounds(model, ii):
        return lwrbnds[ii], model.pwrbase[ii] + model.pwrpeak[ii], uprbnds[ii]

    mm.pwrswithinbnds = pe.Constraint(mm.ind, rule=powerswithinbounds)

    # cyclic constraint startenergy = endenergy
    mm.cyclicbase = pe.Constraint(expr=(mm.startenrgybase ==
                                        mm.enrgybase[mm.ind.last()]))
    mm.cyclicpeak = pe.Constraint(expr=(mm.startenrgypeak ==
                                        mm.enrgypeak[mm.ind.last()]))

    # constraint split power in positive and negative part
    def powerequalbase(model, ii):
        return model.pwrbase[ii] == model.pwrplusbase[ii] + \
                                    model.pwrminusbase[ii]

    def powerequalpeak(model, ii):
        return model.pwrpeak[ii] == model.pwrpluspeak[ii] + \
                                    model.pwrminuspeak[ii]

    mm.pwrequalbase = pe.Constraint(mm.ind, rule=powerequalbase)
    mm.pwrequalpeak = pe.Constraint(mm.ind, rule=powerequalpeak)

    # constraint self discharge and efficiency losses
    def lossesbase(model, ii):
        inter = model.pwrinter[ii]
        eff_losses = (model.pwrplusbase[ii]*efficiency +
                      model.pwrminusbase[ii]/efficiency)
        if ii is 0:
            disch_losses = -model.startenrgybase*disch
        else:
            disch_losses = -model.enrgybase[ii-1]*disch
        return model.pwridealbase[ii] == inter + eff_losses + disch_losses

    def lossespeak(model, ii):
        inter = -model.pwrinter[ii]
        eff_losses = (model.pwrpluspeak[ii]*efficiency +
                      model.pwrminuspeak[ii]/efficiency)
        if ii is 0:
            disch_losses = -model.startenrgypeak*disch
        else:
            disch_losses = -model.enrgypeak[ii-1]*disch
        return model.pwridealpeak[ii] == inter + eff_losses + disch_losses

    mm.pwrafterlossesbase = pe.Constraint(mm.ind, rule=lossesbase)
    mm.pwrafterlossespeak = pe.Constraint(mm.ind, rule=lossespeak)

    # constraint integrate energy - connect power and energy
    def integrate_powerbase(model, ii):
        if ii is 0:
            return model.enrgybase[ii] == model.startenrgybase + \
                                          model.pwridealbase[ii]*dtime
        else:
            return model.enrgybase[ii] == model.enrgybase[ii-1] + \
                                          model.pwridealbase[ii]*dtime

    def integrate_powerpeak(model, ii):
        if ii is 0:
            return model.enrgypeak[ii] == model.startenrgypeak + \
                                          model.pwridealpeak[ii]*dtime
        else:
            return model.enrgypeak[ii] == model.enrgypeak[ii-1] + \
                                          model.pwridealpeak[ii]*dtime

    mm.intpwrbase = pe.Constraint(mm.ind, rule=integrate_powerbase)
    mm.intpwrpeak = pe.Constraint(mm.ind, rule=integrate_powerpeak)

    # objective
    multiplier1 = 1-1e-1
    multiplier2 = 1
    objective_expr = mm.maxenrgypeak + multiplier1*mm.maxenrgybase
    quadratic_expr = sum((mm.pwrbase[ii] + mm.pwrpeak[ii])**2 for ii in mm.ind)
    penalty_expr_base = sum(mm.pwrplusbase[ii] - mm.pwrminusbase[ii]
                            for ii in mm.ind)
    penalty_expr_peak = sum(mm.pwrpluspeak[ii] - mm.pwrminuspeak[ii]
                            for ii in mm.ind)
    quadratic_penalty_base = sum((mm.pwrplusbase[ii] - mm.pwrminusbase[ii] +
                                  mm.pwrpluspeak[ii] - mm.pwrminuspeak[ii])
                                 for ii in mm.ind)
    mm.obj = pe.Objective(expr=(objective_expr +
                                0*quadratic_expr +
                                multiplier2*(0*penalty_expr_base +
                                             0*quadratic_penalty_base +
                                             0*penalty_expr_peak)))

    return mm


def main():
    print('{:%H:%M:%S}: Making Model...'.format(datetime.now()))
    lwr, upr, xx = alt_values()
    model = make_model(lwr, upr)

    print('{:%H:%M:%S}: Setting up Solver...'.format(datetime.now()))
    solver = pe.SolverFactory('gurobi')

    print('{:%H:%M:%S}: Solving Model...'.format(datetime.now()))
    res = solver.solve(model)
    # model.pprint()

    print('{:%H:%M:%S}: Starting Post Processing...'.format(datetime.now()))
    pwrbase = model.pwrbase.get_values().values()
    pwridealbase = model.pwridealbase.get_values().values()
    pwrpeak = model.pwrpeak.get_values().values()
    pwridealpeak = model.pwridealpeak.get_values().values()
    pwrboth = [b + p for p, b in zip(pwrbase, pwrpeak)]
    pwridealboth = [b + p for p, b in zip(pwridealbase, pwridealpeak)]

    plt.figure()
    plt.step(xx, upr, color='k', linestyle='--')
    plt.step(xx, lwr, color='k', linestyle='--')
    plt.step(xx, pwrboth, color='b')
    plt.step(xx, pwridealboth, color='b', linestyle='--')
    plt.step(xx, pwrbase, color='g')
    plt.step(xx, pwridealbase, color='g', linestyle='--')
    plt.step(xx, pwrpeak, color='r')
    plt.step(xx, pwridealpeak, color='r', linestyle='--')
    plt.grid()

    plt.figure()
    plt.step(xx, model.enrgybase.get_values().values(), color='g')
    plt.step(xx, model.enrgypeak.get_values().values(), color='r')
    plt.grid()

    pwrbaseplus = model.pwrplusbase.get_values().values()
    pwrbaseminus = model.pwrminusbase.get_values().values()
    pwrpeakplus = model.pwrpluspeak.get_values().values()
    pwrpeakminus = model.pwrminuspeak.get_values().values()
    diffbase = [b + p - s for b, p, s in
                zip(pwrbaseplus, pwrbaseminus, pwrbase)]
    diffpeak = [b + p - s for b, p, s in
                zip(pwrpeakplus, pwrpeakminus, pwrpeak)]

    plt.figure()
    plt.step(xx, diffbase, color='g')
    plt.step(xx, diffpeak, color='r')
    plt.grid()
    return model, res


def make_hybridcurve():
    cuts = np.array([0.15, 0.3, 0.5, 0.75])
    pbase = cuts*2
    ppeak = (1 - cuts)*2
    for cut, pb, pp in zip(cuts, pbase, ppeak):
        print('cut = {}'.format(cut))
        print(4*' ' + 'pbase = {}, ppeak = {}'.format(pb, pp))
        lwr, upr, _ = alt_values()
        model = make_model(lwr, upr, maxpwr_base=pb, maxpwr_peak=pp)
        solver = pe.SolverFactory('gurobi')
        solver.solve(model)
        ebase = list(model.maxenrgybase.get_values().values())[0]
        epeak = list(model.maxenrgypeak.get_values().values())[0]
        print(4*' ' + 'etotal = {}'.format(ebase + epeak))
        print(4*' ' + 'ebase = {}, epeak = {}'.format(ebase, epeak))


if __name__ == '__main__':
    starttime = time.time()
    # MODEL, RES = main()
    make_hybridcurve()
    print('Eleapsed time is {:1.4f}s'.format(time.time() - starttime))
