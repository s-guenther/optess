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
import scipy.linalg as sclinalg
import matplotlib.pyplot as plt
import time

from opt_generic import std_values, alt_values


def check_lower_upper(lower, upper):
    """Checks if lower is smaller than upper at all points"""
    if not all(lwr <= upr for lwr, upr in zip(lower, upper)):
        raise ValueError('lower > upper at at least one point')


def make_model(lower, upper, efficiency=1, disch=1e-10, maxpwr_base=1,
               maxpwr_peak=1, dtime=1, **kwargs):
    """Builds a pyomo model, taking lower and upper as input and minimizes
    storage dimension, for 2 storage system"""
    check_lower_upper(lower, upper)

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
    mm.pwrinter = pe.Var(mm.ind, bounds=(-min(maxpwr_base, maxpwr_peak),
                                         min(maxpwr_base, maxpwr_peak)))
    # set this to zero to prevent reloading
    # mm.pwrinter = pe.Var(mm.ind, bounds=(0, 0))

    def interandbasewithinlimits(model, ii):
        return (-maxpwr_base,
                model.pwrbase[ii] + model.pwrinter[ii],
                maxpwr_base)

    def interandpeakwithinlimits(model, ii):
        return (-maxpwr_peak,
                model.pwrpeak[ii] - model.pwrinter[ii],
                maxpwr_peak)

    mm.baseinterwithinbnds = pe.Constraint(mm.ind,
                                           rule=interandbasewithinlimits)
    mm.peakinterwithinbnds = pe.Constraint(mm.ind,
                                           rule=interandpeakwithinlimits)

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
        return model.pwrbase[ii] == \
               model.pwrplusbase[ii] + model.pwrminusbase[ii]

    def powerequalpeak(model, ii):
        return model.pwrpeak[ii] == \
               model.pwrpluspeak[ii] + model.pwrminuspeak[ii]

    mm.pwrequalbase = pe.Constraint(mm.ind, rule=powerequalbase)
    mm.pwrequalpeak = pe.Constraint(mm.ind, rule=powerequalpeak)

    # constraint self discharge and efficiency losses
    def lossesbase(model, ii):
        eff_losses = (model.pwrplusbase[ii] * efficiency +
                      model.pwrminusbase[ii] / efficiency)
        if ii is 0:
            disch_losses = -model.startenrgybase * disch
        else:
            disch_losses = -model.enrgybase[ii - 1] * disch
        return model.pwridealbase[ii] == eff_losses + disch_losses

    def lossespeak(model, ii):
        eff_losses = (model.pwrpluspeak[ii] * efficiency +
                      model.pwrminuspeak[ii] / efficiency)
        if ii is 0:
            disch_losses = -model.startenrgypeak * disch
        else:
            disch_losses = -model.enrgypeak[ii - 1] * disch
        return model.pwridealpeak[ii] == eff_losses + disch_losses

    mm.pwrafterlossesbase = pe.Constraint(mm.ind, rule=lossesbase)
    mm.pwrafterlossespeak = pe.Constraint(mm.ind, rule=lossespeak)

    # constraint integrate energy - connect power and energy
    def integrate_powerbase(model, ii):
        if ii is 0:
            lastenergy = model.startenrgybase
        else:
            lastenergy = model.enrgybase[ii - 1]
        return model.enrgybase[ii] == lastenergy + \
               (model.pwrinter[ii] + model.pwridealbase[ii]) * dtime

    def integrate_powerpeak(model, ii):
        if ii is 0:
            lastenergy = model.startenrgypeak
        else:
            lastenergy = model.enrgypeak[ii - 1]
        return model.enrgypeak[ii] == lastenergy + \
               (-model.pwrinter[ii] + model.pwridealpeak[ii]) * dtime

    mm.intpwrbase = pe.Constraint(mm.ind, rule=integrate_powerbase)
    mm.intpwrpeak = pe.Constraint(mm.ind, rule=integrate_powerpeak)

    # objective
    multiplier1 = 1 - 1e-1
    multiplier2 = 1
    objective_expr = mm.maxenrgypeak + multiplier1 * mm.maxenrgybase
    quadratic_expr = sum(
        (mm.pwrbase[ii] + mm.pwrpeak[ii]) ** 2 for ii in mm.ind)
    penalty_expr_base = sum(mm.pwrplusbase[ii] - mm.pwrminusbase[ii]
                            for ii in mm.ind)
    penalty_expr_peak = sum(mm.pwrpluspeak[ii] - mm.pwrminuspeak[ii]
                            for ii in mm.ind)
    quadratic_penalty_base = sum((mm.pwrplusbase[ii] - mm.pwrminusbase[ii] +
                                  mm.pwrpluspeak[ii] - mm.pwrminuspeak[ii])
                                 for ii in mm.ind)
    mm.obj = pe.Objective(expr=(objective_expr +
                                0 * quadratic_expr +
                                multiplier2 * (1 * penalty_expr_base +
                                               0 * quadratic_penalty_base +
                                               1 * penalty_expr_peak)))

    return mm


def is_valid_result(result, rel_tol=1e-9):
    """Checks if the optimization results are plausible"""
    # TODO only rudimentary check
    res = result
    diffbase = [plus + minus - added for plus, minus, added in
                zip(res['baseplus'], res['baseminus'], res['base'])]
    diffpeak = [plus + minus - added for plus, minus, added in
                zip(res['peakplus'], res['peakminus'], res['peak'])]
    normdiff = sclinalg.norm(diffbase) + sclinalg.norm(diffpeak)
    normpwr = (sclinalg.norm(list(res['base'])) +
               sclinalg.norm(list(res['peak'])))
    return normdiff/normpwr <= rel_tol


def extract_results(model, xvals=None):
    """Extracts neccessary information from solved model"""
    res = dict()
    # Variables of model

    res['base'] = model.pwrbase.get_values().values()
    res['peak'] = model.pwrpeak.get_values().values()
    res['inter'] = model.pwrinter.get_values().values()

    res['idealbase'] = model.pwridealbase.get_values().values()
    res['idealpeak'] = model.pwridealpeak.get_values().values()

    if xvals is None:
        res['xvals'] = list(range(len(res['base'])))
    else:
        res['xvals'] = xvals

    res['baseplus'] = model.pwrplusbase.get_values().values()
    res['baseminus'] = model.pwrminusbase.get_values().values()
    res['peakplus'] = model.pwrpluspeak.get_values().values()
    res['peakminus'] = model.pwrminuspeak.get_values().values()

    res['energybase'] = model.enrgybase.get_values().values()
    res['maxenergybase'] = model.maxenrgybase.get_values().values()
    res['energypeak'] = model.enrgypeak.get_values().values()
    res['maxenergypeak'] = model.maxenrgypeak.get_values().values()
    res['startenergybase'] = model.startenrgybase.get_values().values()
    res['startenergypeak'] = model.startenrgypeak.get_values().values()

    # Derived variables
    res['both'] = [b + p for p, b in zip(res['base'], res['peak'])]
    res['idealboth'] = [b + p for p, b in
                        zip(res['idealbase'], res['idealpeak'])]

    res['interbase'] = res['inter']
    res['interbaseplus'] = [v if v >= 0 else 0 for v in res['interbase']]
    res['interbaseminus'] = [v if v < 0 else 0 for v in res['interbase']]

    res['interpeak'] = [-val for val in res['interbase']]
    res['interpeakplus'] = [v if v >= 0 else 0 for v in res['interpeak']]
    res['interpeakminus'] = [v if v < 0 else 0 for v in res['interpeak']]

    res['idealbaseplus'] = [v if v >= 0 else 0 for v in res['idealbase']]
    res['idealbaseminus'] = [v if v < 0 else 0 for v in res['idealbase']]
    res['idealpeakplus'] = [v if v >= 0 else 0 for v in res['idealpeak']]
    res['idealpeakminus'] = [v if v < 0 else 0 for v in res['idealpeak']]

    return res


def is_solverstatus_ok(solverres):
    """Extracts status message of pyomo optimization result object"""
    # TODO implement me
    return True


def prepare_plot(xvals, lower, upper, maxpwr_base=0.5, maxpwr_peak=1.5,
                 fig=100, **kwargs):
    """Prepares a figure for a specific problem defined by its upper and
    lower bounds and the storage dimensions (draws some background bounds)"""

    check_lower_upper(lower, upper)

    plt.figure(fig)
    plt.clf()

    plt.plot(xvals, [0] * len(lower), color='k')
    # add limits of added storages
    lowerboth = list(max(lwr, -(maxpwr_base + maxpwr_peak)) for lwr in lower)
    upperboth = list(min(upr, (maxpwr_base + maxpwr_peak)) for upr in upper)
    plt.fill_between(xvals, lowerboth, upperboth, color=[0.85]*3, step='pre')
    plt.grid()
    # add single storage limits - positive side
    maxstor = max(maxpwr_peak, maxpwr_base)
    minstor = min(maxpwr_peak, maxpwr_base)
    lowerstorpos = list(max(minstor, lwr) for lwr in lower)
    upperstorpos = list(min(maxstor, upr) for upr in upper)
    truevalspos = list(lwr <= upr for lwr, upr
                       in zip(lowerstorpos, upperstorpos))
    plt.fill_between(xvals, lowerstorpos, upperstorpos, where=truevalspos,
                     color=[0.8] * 3, step='pre')
    # add single storage limits - negative side
    lowerstorneg = list(max(-maxstor, lwr) for lwr in lower)
    upperstorneg = list(min(-minstor, upr) for upr in upper)
    truevalsneg = list(lwr <= upr for lwr, upr
                       in zip(lowerstorneg, upperstorneg))
    plt.fill_between(xvals, lowerstorneg, upperstorneg, where=truevalsneg,
                     color=[0.8] * 3, step='pre')


def add_results_to_plot(result, fig=100):
    """adds peak base and added power profiles to available plot"""

    res = result
    xvals = res['xvals']

    plt.figure(fig)
    plt.step(xvals, res['idealboth'], color='b')
    # plt.step(xvals, res['inter'], color='k')
    plt.stackplot(xvals,
                  res['idealbaseplus'], res['idealpeakplus'],
                  res['interbaseplus'], res['interpeakplus'],
                  step='pre',
                  colors=('green', 'red', 'limegreen', 'orangered'))
    plt.stackplot(xvals,
                  res['idealbaseminus'], res['idealpeakminus'],
                  res['interbaseminus'], res['interpeakminus'],
                  step='pre',
                  colors=('green', 'red', 'limegreen', 'orangered'))

    plt.figure(fig+1)
    plt.clf()
    plt.step(xvals, res['energybase'], color='g')
    plt.step(xvals, res['energypeak'], color='r')
    plt.grid()


def plot_case(data, result):
    """Wrapper function combinging prepare_plot and add_results_to_plot"""
    prepare_plot(**data)
    add_results_to_plot(result, fig=data['fig'])


def main(profile=None):
    """Solve a model and plot it (use default if no data is provided)"""
    if profile is None:
        profile = alt_values

    print('{:%H:%M:%S}: Making Model...'.format(datetime.now()))
    lwr, upr, xx = profile()
    data = dict(lower=lwr,
                upper=upr,
                efficiency=0.99,
                disch=1e-3,
                maxpwr_base=1,
                maxpwr_peak=1,
                xvals=xx,
                dtime=xx[1] - xx[0],
                fig=100)
    model = make_model(**data)

    print('{:%H:%M:%S}: Setting up Solver...'.format(datetime.now()))
    solver = pe.SolverFactory('gurobi')

    print('{:%H:%M:%S}: Solving Model...'.format(datetime.now()))
    pyomo_res = solver.solve(model)
    if not is_solverstatus_ok(pyomo_res):
        print('\nCAUTION: SOLVER APPARENTLY DID NOT CONVERGE OR MODEL WAS '
              'PROVEN TO BE INFEASIBLY.\n')

    result = extract_results(model, data['xvals'])
    if not is_valid_result(result, rel_tol=1e-1):
        print('\nCAUTION: RESULT SEEMS TO BE INVALID\n')

    print('{:%H:%M:%S}: Starting Post Processing...'.format(datetime.now()))
    plot_case(data, result)

    return model, result


def make_hybridcurve():
    cuts = np.array([0.15, 0.3, 0.5, 0.75])
    pbase = cuts * 2
    ppeak = (1 - cuts) * 2
    for cut, pb, pp in zip(cuts, pbase, ppeak):
        print('cut = {}'.format(cut))
        print(4 * ' ' + 'pbase = {}, ppeak = {}'.format(pb, pp))
        lwr, upr, _ = alt_values()
        model = make_model(lwr, upr, maxpwr_base=pb, maxpwr_peak=pp)
        solver = pe.SolverFactory('gurobi')
        solver.solve(model)
        ebase = list(model.maxenrgybase.get_values().values())[0]
        epeak = list(model.maxenrgypeak.get_values().values())[0]
        print(4 * ' ' + 'etotal = {}'.format(ebase + epeak))
        print(4 * ' ' + 'ebase = {}, epeak = {}'.format(ebase, epeak))


if __name__ == '__main__':
    starttime = time.time()
    MODEL, RES = main()
    # make_hybridcurve()
    print('Elapsed time is {:1.4f}s'.format(time.time() - starttime))
