#!/usr/bin/env python3
"""This module aggregates helper functions related to module 'model'"""

import pyomo.environ as pe


# ###
# ### Wrapper functions
# ###

def add_vars(data, model):
    """Defines all Variables of the optimization model"""
    # Extract Variables from Data
    signal = data.signal
    base = data.base
    peak = data.peak

    # Set for range of variables
    model.ind = pe.Set(initialize=range(len(signal)), ordered=True)

    # Create empty objective expression, which is filled in subsequent steps
    model.objexpr = 0

    _add_power_vars(model, base, peak)
    _add_energy_vars(model)
    _add_binary_vars(model)


def add_constraints(data, model):
    """Defines all Constraints of the optimization model"""
    # Extract Variables from data
    signal = data.signal
    base = data.base
    peak = data.peak

    _lock_plus_and_minus_constraint(model)
    _bounds_for_inter_constraint(model, base, peak)
    _loss_model_constraint(model, signal, base, peak)
    _integrate_power_constraint(model, signal)
    _cyclic_energy_constraint(model)
    _binary_bounds_constraint(model, base, peak)
    _binary_interval_locks(model)
    _binary_cross_locks(model)


def add_inter_constraint():
    """Adds constraint that allows an inter-storage power flow. This
    function does nothing as it is automatically satisfied, but is included
    for symmetry"""
    pass


def add_nointer_constraint(model):
    """Sets inter storage power flow to zero to prevent reloading"""
    def nointer_constraint(mod, ii):
        return mod.inter[ii] == 0

    def bin_nointer_constraint(mod, ii):
        return mod.bininterlower[ii] + mod.bininterupper[ii] == 0

    model.con_nointer = pe.Constraint(model.ind, rule=nointer_constraint)
    model.con_bin_nointer = pe.Constraint(model.ind,
                                          rule=bin_nointer_constraint)


def add_capacity_minimizing_objective(model, multiplier):
    """Adds objective function that minimizes energy capacity of peak and
    base"""
    model.objexpr += model.peakenergycapacity
    model.objexpr += multiplier*model.baseenergycapacity
    model.obj = pe.Objective(expr=model.objexpr)


def add_peak_cutting_objective(data, model):
    """Add objective - this is an objective or aim in a larger sense as it
    will cut peak power which also adds constraints to reach the objective."""
    signal = data.signal.value
    minpower = data.objective.value.min
    maxpower = data.objective.value.max

    def cutting_low(mod, ii):
        return signal[ii] - (mod.base[ii] + mod.peak[ii]) >= minpower

    def cutting_high(mod, ii):
        return signal[ii] - (mod.base[ii] + mod.peak[ii]) <= maxpower

    model.con_cutting_low = pe.Constraint(model.ind, rule=cutting_low)
    model.con_cutting_high = pe.Constraint(model.ind, rule=cutting_high)


def add_throughput_objective(data, model):
    """Add objective - this is an objective or aim in a larger sense as it
    will decrease the amount of energy taken from grid/supply etc. It will
    also add constraints to reach objective. This objective only makes sense
    if the power from grid changes sign, i.e. if power is fed into grid"""
    signal = data.signal.value
    dtime = data.signal.dtime
    maxenergy = data.objective.value

    model.deltaplus = pe.Var(model.ind, bounds=(0, None))
    model.deltaminus = pe.Var(model.ind, bounds=(None, 0))

    def split_delta(mod, ii):
        return (signal[ii] - (mod.base[ii] + mod.peak[ii]) <=
                mod.deltaplus[ii] + mod.deltaminus[ii])

    model.con_split_delta = pe.Constraint(model.ind, rule=split_delta)

    model.con_throughput = \
        pe.Constraint(expr=sum(model.deltaplus[ii]*dtime[ii] <= maxenergy
                               for ii in model.ind))


# ###
# ### Elementary Functions
# ###

def _add_power_vars(model, base, peak):
    # Base power and peak power
    model.base = pe.Var(model.ind, bounds=(base.power.min, base.power.max))
    model.peak = pe.Var(model.ind, bounds=(peak.power.min, peak.power.max))

    # Positive and negative part of base and peak power
    model.baseplus = pe.Var(model.ind, bounds=(0, base.power.max))
    model.baseminus = pe.Var(model.ind, bounds=(base.power.min, 0))
    model.peakplus = pe.Var(model.ind, bounds=(0, peak.power.max))
    model.peakminus = pe.Var(model.ind, bounds=(peak.power.min, 0))

    # Ideal inner base and peak power considering losses
    model.baseinner = pe.Var(model.ind)
    model.peakinner = pe.Var(model.ind)

    # Power for inter-storage power flow
    model.inter = pe.Var(model.ind,
                         bounds=(max(base.power.min, peak.power.max),
                                 min(base.power.max, base.power.max)))


def _add_energy_vars(model):
    # Energy content as function of time
    model.baseenergy = pe.Var(model.ind, bounds=(0, None))
    model.peakenergy = pe.Var(model.ind, bounds=(0, None))

    # Energy capacity (maximum storable energy)
    model.baseenergycapacity = pe.Var(bounds=(0, None))
    model.peakenergycapacity = pe.Var(bounds=(0, None))

    # Initial condition of base and peak energy content
    model.baseenergyinit = pe.Var(bounds=(0, None))
    model.peakenergyinit = pe.Var(bounds=(0, None))


def _add_binary_vars(model):
    # Binary variable for base lower and upper bound switch
    model.binbaselower = pe.Var(within=pe.Binary)
    model.binbaseupper = pe.Var(within=pe.Binary)

    # Binary variable for peak lower and upper bound switch
    model.binpeaklower = pe.Var(within=pe.Binary)
    model.binpeakupper = pe.Var(within=pe.Binary)

    # Binary variable for inter storage power lower and upper bound switch
    model.bininterlower = pe.Var(within=pe.Binary)
    model.bininterupper = pe.Var(within=pe.Binary)


def _lock_plus_and_minus_constraint(model, multiplier=1):
    """ensure that powerplus + powerminus = power, this also adds a soft
    constraint to the objective expression, minimizing powerplus - powerminus
    to prevent something like   power = plus + minus <==> 5 = 7 + -2
    and gain                    power = plus + minus <==> 5 = 5 + 0 instead"""

    def lock_base(mod, ii):
        return mod.base[ii] == mod.baseplus[ii] + mod.baseminus[ii]

    def lock_peak(mod, ii):
        return mod.peak[ii] == mod.peakplus[ii] + mod.peakminus[ii]

    model.con_lockbase = pe.Constraint(model.ind, rule=lock_base)
    model.con_lockpeak = pe.Constraint(model.ind, rule=lock_peak)

    model.objexpr += (model.baseplus - model.baseminus +
                      model.peakplus - model.peakminus)*multiplier


def _bounds_for_inter_constraint(model, base, peak):
    """Add a constraints which acts as bound, limiting the value of inter
    power in a way that the storage power capacities are not exceeded."""

    def bounds_inter_base(mod, ii):
        return base.power.min, mod.base[ii] + mod.inter[ii], base.power.max

    def bounds_inter_peak(mod, ii):
        return peak.power.min, mod.peak[ii] - mod.inter[ii], peak.power.max

    model.con_boundsinterbase = pe.Constraint(model.ind,
                                              rule=bounds_inter_base)
    model.con_boundsinterpeak = pe.Constraint(model.ind,
                                              rule=bounds_inter_peak)


def _loss_model_constraint(model, signal, base, peak):
    """Implements efficiency losses and self discharge losses and links base
    power with inner base power (and peak power respectively)"""

    dtime = signal.dtime

    def base_losses(mod, ii):
        efficiency_losses = (mod.baseplus[ii] * base.efficiency.charge +
                             mod.baseminus[ii] / base.efficiency.discharge)
        if ii is 0:
            discharge_losses = -mod.baseenergyinit * \
                               base.selfdischarge*dtime[ii]
        else:
            discharge_losses = -mod.baseenergy[ii - 1] * \
                               base.selfdischarge*dtime[ii]
        return mod.baseinner[ii] == efficiency_losses + discharge_losses

    def peak_losses(mod, ii):
        efficiency_losses = (mod.peakplus[ii] * peak.efficiency.charge +
                             mod.peakminus[ii] / peak.efficiency.discharge)
        if ii is 0:
            discharge_losses = -mod.peakenergyinit * \
                               peak.selfdischarge*dtime[ii]
        else:
            discharge_losses = -mod.peakenergy[ii - 1] * \
                               peak.selfdischarge*dtime[ii]
        return mod.peakinner[ii] == efficiency_losses + discharge_losses

    model.con_baselosses = pe.Constraint(model.ind, rule=base_losses)
    model.con_peaklosses = pe.Constraint(model.ind, rule=peak_losses)


def _integrate_power_constraint(model, signal):
    """Euler Integration per timestep of power to gain energy"""

    dtime = signal.dtime

    # constraint integrate energy - connect power and energy
    def integrate_base(mod, ii):
        if ii is 0:
            lastenergy = mod.baseenergyinit
        else:
            lastenergy = mod.baseenergy[ii - 1]
        return (mod.baseenergy[ii] == lastenergy +
                (mod.inter[ii] + mod.baseinner[ii]) * dtime[ii])

    def integrate_peak(mod, ii):
        if ii is 0:
            lastenergy = mod.peakenergyinit
        else:
            lastenergy = mod.peakenergy[ii - 1]
        return (mod.peakenergy[ii] == lastenergy +
                (mod.inter[ii] + mod.peakinner[ii]) * dtime[ii])

    model.con_integratebase = pe.Constraint(model.ind, rule=integrate_base)
    model.con_integratepeak = pe.Constraint(model.ind, rule=integrate_peak)


def _cyclic_energy_constraint(model):
    """Ensures that energy content at beginning of interval equals energy
    content at end of interval"""

    model.con_cyclicbase = \
        pe.Constraint(expr=(model.baseenergyinit ==
                            model.baseenergy[model.ind.last()]))
    model.con_cyclicpeak = \
        pe.Constraint(expr=(model.peakenergyinit ==
                            model.peakenergy[model.ind.last()]))


def _energy_lower_max_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""
    def energy_base_lower_max(mod, ii):
        return mod.baseenergy[ii] <= mod.basenergycapacity

    def energy_peak_lower_max(mod, ii):
        return mod.baseenergy[ii] <= mod.peakenergycapacity

    model.con_baseenergylowermax = pe.Constraint(model.ind,
                                                 rule=energy_base_lower_max)
    model.con_peakenergylowermax = pe.Constraint(model.ind,
                                                 rule=energy_peak_lower_max)

    model.con_baseenergyinitlowermax = \
        pe.Constraint(expr=(model.baseenergyinit <= model.baseenergycapacity))
    model.con_peakenergyinitlowermax = \
        pe.Constraint(expr=(model.peakenergyinit <= model.peakenergycapacity))


def _binary_bounds_constraint(model, base, peak):
    """Buts a binary variable on all relevant power bounds"""

    def bin_bound_base(mod, ii):
        return (base.power.min*mod.binbaselower[ii],
                mod.base[ii],
                base.power.max*mod.binbaseupper[ii])

    def bin_bound_peak(mod, ii):
        return (peak.power.min*mod.binpeaklower[ii],
                mod.peak[ii],
                peak.power.max*mod.binpeakupper[ii])

    inter_minpower = max(base.power.min, peak.power.min)
    inter_maxpower = min(base.power.max, peak.power.ax)

    def bin_bound_inter(mod, ii):
        return (inter_minpower*mod.bininterlower[ii],
                mod.base[ii],
                inter_maxpower*mod.bininterupper[ii])

    model.con_bin_bound_base = pe.Constraint(model.ind, rule=bin_bound_base)
    model.con_bin_bound_peak = pe.Constraint(model.ind, rule=bin_bound_peak)
    model.con_bin_bound_inter = pe.Constraint(model.ind, rule=bin_bound_inter)


def _binary_interval_locks(model):
    """Connects binary variables to ensure that only one direction of real
    axis for base, peak and inter is opened"""

    def bin_base_interval(mod, ii):
        """Ensures that base bounds are opened to either positive real axis
        or negative real axes"""
        return mod.binbaselower[ii] + mod.binbaseupper[ii] <= 1

    def bin_peak_interval(mod, ii):
        """Ensures that peak bounds are opened to either positive real axis
        or negative real axes"""
        return mod.binpeaklower[ii] + mod.binpeakupper[ii] <= 1

    def bin_inter_interval(mod, ii):
        """Ensures that inter bounds are opened to either positive real axis
        or negative real axes"""
        return mod.bininterlower[ii] + mod.bininterupper[ii] <= 1

    model.con_bin_base_interval = pe.Constraint(model.ind,
                                                rule=bin_base_interval)
    model.con_bin_peak_interval = pe.Constraint(model.ind,
                                                rule=bin_peak_interval)
    model.con_bin_inter_interval = pe.Constraint(model.ind,
                                                 rule=bin_inter_interval)


def _binary_cross_locks(model):
    """Connects left peak with right right base and so on"""

    def cross_base_peak(mod, ii):
        return mod.binbaseupper[ii] + mod.binpeaklower[ii] <= 1

    def cross_peak_base(mod, ii):
        return mod.binpeakupper[ii] + mod.binbaselower[ii] <= 1

    def cross_base_inter(mod, ii):
        return mod.binbaseupper[ii] + mod.bininterlower[ii] <= 1

    def cross_inter_base(mod, ii):
        return mod.bininterupper[ii] + mod.binbaselower[ii] <= 1

    def cross_peak_inter(mod, ii):
        return mod.binpeakupper[ii] + mod.bininterlower[ii] <= 1

    def cross_inter_peak(mod, ii):
        return mod.bininterupper[ii] + mod.binpeaklower[ii] <= 1

    model.con_cross_base_peak = pe.Constraint(model.ind,
                                              rule=cross_base_peak)
    model.con_cross_peak_base = pe.Constraint(model.ind,
                                              rule=cross_peak_base)

    model.con_cross_base_inter = pe.Constraint(model.ind,
                                               rule=cross_base_inter)
    model.con_cross_inter_base = pe.Constraint(model.ind,
                                               rule=cross_inter_base)

    model.con_cross_peak_inter = pe.Constraint(model.ind,
                                               rule=cross_peak_inter)
    model.con_cross_inter_peak = pe.Constraint(model.ind,
                                               rule=cross_inter_peak)
