#!/usr/bin/env python3
"""This module aggregates helper functions related to module 'optimmodel'
Its related to building the pyomo model for HESS"""

import pyomo.environ as pe
from copy import deepcopy

from .objective import Objective, Strategy
from .signal import Signal
from .storage import Storage


class HybridBuilder:
    def __init__(self):
        self.signal = None
        self.base = None
        self.peak = None
        self.objective = None
        self.strategy = None
        self.model = None
        self.model_2nd = None

    ###########################################################################
    # noinspection PyArgumentList
    def minimize_energy(self, signal, base, peak, objective,
                        strategy=Strategy.inter, name='Hybrid EES Model'):
        self.signal = Signal(signal)
        self.base = Storage(base)
        self.peak = Storage(peak)
        self.objective = Objective(objective)
        self.strategy = Strategy(strategy)

        self.model = pe.ConcreteModel(name=name)

        self._add_vars()
        self._add_constraints()

        if strategy == Strategy.inter:
            self._add_inter_constraint()
        else:
            self._add_nointer_constraint()

        if objective.type.name == 'power':
            self._add_peak_cutting_objective()
        else:
            self._add_throughput_objective()

        self._add_capacity_minimizing_objective()

        return self.model

    ###########################################################################
    def minimize_cycles(self, baseenergy=None, peakenergy=None,
                        multiplier=1e-1, name='Hybrid EES Model Cycle '
                                              'Minimization'):
        self.model_2nd = deepcopy(self.model)
        model = self.model_2nd
        model.name = name

        # Fix previously determined base and peak energy capacity
        if not baseenergy:
            baseenergy = \
                list(model.baseenergycapacity.get_values().values())[0]
        if not peakenergy:
            peakenergy = \
                list(model.peakenergycapacity.get_values().values())[0]
        self.model_2nd.con_lock_baseenergy_capacity = \
            pe.Constraint(expr=model.baseenergycapacity == baseenergy)
        self.model_2nd.con_lock_peakenergy_capacity = \
            pe.Constraint(expr=model.peakenergycapacity == peakenergy)

        # Add quadratic minimizing expression
        model.interplus = pe.Var(model.ind, bounds=(0, None))
        model.interminus = pe.Var(model.ind, bounds=(None, 0))

        model.con_lockinter = pe.Constraint(model.ind, rule=_lock_inter)

        # Define a monotonically incrasing vector which will be multiplied
        # with interminus. This way, interstorage power flow becomes
        # determined and unambiguous (hopefully)
        monoinc = [(model.ind.last() + ii) for ii in model.ind]

        model.objexpr = sum((-model.baseminus[ii]
                             + (model.interplus[ii] -
                                model.interminus[ii])
                             - model.peakminus[ii]*multiplier)*monoinc[ii]
                            for ii in model.ind)
        model.del_component(model.obj)
        model.obj = pe.Objective(expr=model.objexpr)

        return model

    ###########################################################################
    def build(self, signal, base, peak, objective, strategy, name):
        return self.minimize_energy(signal, base, peak,
                                    objective, strategy, name)

    ###########################################################################
    def _add_vars(self):
        """Defines all Variables of the optimization model"""
        # Extract Variables from Data
        signal = self.signal
        base = self.base
        peak = self.peak
        objective = self.objective
        model = self.model

        # Set for range of variables
        self.model.ind = pe.Set(initialize=range(len(signal)), ordered=True)

        # Create empty objective expression, which is filled in subsequent
        # steps
        self.model.objexpr = 0

        _add_aux_vars(model, signal, base, peak, objective)
        _add_power_vars(model)
        _add_energy_vars(model)
        _add_binary_vars(model)

    ###########################################################################
    def _add_constraints(self):
        """Defines all Constraints of the optimization model"""
        # Extract Variables from data
        model = self.model

        _lock_plus_and_minus_constraint(model)
        _bounds_for_inter_constraint(model)
        _loss_model_constraint(model)
        _integrate_power_constraint(model)
        _cyclic_energy_constraint(model)
        _energy_lower_max_constraint(model)
        _binary_bounds_constraint(model)
        _binary_interval_locks(model)
        _binary_cross_locks(model)
        # _binary_number_locks(model)

    ###########################################################################
    def _add_inter_constraint(self):
        """Adds constraint that allows an inter-storage power flow. This
        function does nothing as it is automatically satisfied, but is included
        for symmetry"""
        pass

    ###########################################################################
    def _add_nointer_constraint(self):
        """Sets inter storage power flow to zero to prevent reloading"""
        model = self.model

        model.con_nointer = \
            pe.Constraint(model.ind, rule=_nointer_constraint)
        model.con_bin_nointer = \
            pe.Constraint(model.ind, rule=_bin_nointer_constraint)

    ###########################################################################
    def _add_capacity_minimizing_objective(self, multiplier=0.50):
        """Adds objective function that minimizes energy capacity of peak and
        base, add this objective last"""
        model = self.model
        model.objexpr += multiplier*model.baseenergycapacity
        model.objexpr += model.peakenergycapacity
        model.obj = pe.Objective(expr=model.objexpr)

    ###########################################################################
    def _add_peak_cutting_objective(self):
        """Add objective - this is an objective or aim in a larger sense as it
        will cut peak power which also adds constraints to reach the
        objective."""
        model = self.model

        model.con_cutting_low = pe.Constraint(model.ind,
                                              rule=_cutting_low)
        model.con_cutting_high = pe.Constraint(model.ind,
                                               rule=_cutting_high)

    ###########################################################################
    def _add_throughput_objective(self):
        """Add objective - this is an objective or aim in a larger sense as it
        will decrease the amount of energy taken from grid/supply etc. It will
        also add constraints to reach objective. This objective only makes
        sense if the power from grid changes sign, i.e. if power is fed into
        grid"""
        dtime = self.signal.dtimes
        maxenergy = self.objective.val.max
        model = self.model

        model.deltaplus = pe.Var(model.ind, bounds=(0, None))
        model.deltaminus = pe.Var(model.ind, bounds=(None, 0))

        model.con_split_delta = \
            pe.Constraint(model.ind, rule=_split_delta)

        model.con_throughput = \
            pe.Constraint(expr=sum(model.deltaplus[ii]*dtime[ii]
                                   for ii in model.ind) <= maxenergy)


###########################################################################
# minimize_cycles(self, baseenergy=None, peakenergy=None,
def _lock_inter(mod, ii):
    return mod.inter[ii] == mod.interplus[ii] + mod.interminus[ii]


###########################################################################
# _add_nointer_constraint(self):
def _nointer_constraint(mod, ii):
    return mod.inter[ii] == 0


def _bin_nointer_constraint(mod, ii):
    return mod.bininterlower[ii] + mod.bininterupper[ii] == 0


###########################################################################
# _add_peak_cutting_objective(self):
# noinspection PyProtectedMember
def _cutting_low(mod, ii):
    signal = mod._signal.vals
    minpower = mod._objective.val.min
    return signal[ii] + (mod.base[ii] + mod.peak[ii]) >= minpower


# noinspection PyProtectedMember
def _cutting_high(mod, ii):
    signal = mod._signal.vals
    maxpower = mod._objective.val.max
    return signal[ii] + (mod.base[ii] + mod.peak[ii]) <= maxpower


###########################################################################
# _add_throughput_objective(self):
def _split_delta(mod, ii):
    # noinspection PyProtectedMember
    signal = mod._signal.vals
    return (signal[ii] + (mod.base[ii] + mod.peak[ii]) <=
            mod.deltaplus[ii] + mod.deltaminus[ii])


# ###
# ### Elementary Functions
# ###

###############################################################################
def _add_aux_vars(model, signal, base, peak, objective):
    model._signal = signal
    model._base = base
    model._peak = peak
    model._objective = objective


###############################################################################
# noinspection PyProtectedMember
def _add_power_vars(model):
    base = model._base
    peak = model._peak
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
                         bounds=(max(base.power.min, -peak.power.max),
                                 min(base.power.max, -peak.power.min)))


###############################################################################
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


###############################################################################
def _add_binary_vars(model):
    # Binary variable for base lower and upper bound switch
    model.binbaselower = pe.Var(model.ind, within=pe.Binary)
    model.binbaseupper = pe.Var(model.ind, within=pe.Binary)

    # Binary variable for peak lower and upper bound switch
    model.binpeaklower = pe.Var(model.ind, within=pe.Binary)
    model.binpeakupper = pe.Var(model.ind, within=pe.Binary)

    # Binary variable for inter storage power lower and upper bound switch
    model.bininterlower = pe.Var(model.ind, within=pe.Binary)
    model.bininterupper = pe.Var(model.ind, within=pe.Binary)


###############################################################################
def __lock_base(mod, ii):
    return mod.base[ii] == mod.baseplus[ii] + mod.baseminus[ii]


def __lock_peak(mod, ii):
    return mod.peak[ii] == mod.peakplus[ii] + mod.peakminus[ii]


def _lock_plus_and_minus_constraint(model, multiplier=1):
    """ensure that powerplus + powerminus = power, this also adds a penalty
    term to the objective expression, minimizing powerplus - powerminus
    to prevent something like   power = plus + minus <==> 5 = 7 + -2
    and gain                    power = plus + minus <==> 5 = 5 + 0 instead"""

    model.con_lockbase = pe.Constraint(model.ind, rule=__lock_base)
    model.con_lockpeak = pe.Constraint(model.ind, rule=__lock_peak)

    model.objexpr += sum(model.baseplus[ii] - model.baseminus[ii] +
                         model.peakplus[ii] - model.peakminus[ii]
                         for ii in model.ind)*multiplier


###############################################################################
def __bounds_inter_base(mod, ii):
    # noinspection PyProtectedMember
    base = mod._base
    return base.power.min, mod.base[ii] + mod.inter[ii], base.power.max


def __bounds_inter_peak(mod, ii):
    # noinspection PyProtectedMember
    peak = mod._peak
    return peak.power.min, mod.peak[ii] - mod.inter[ii], peak.power.max


def _bounds_for_inter_constraint(model):
    """Add a constraints which acts as bound, limiting the value of inter
    power in a way that the storage power capacities are not exceeded."""
    model.con_boundsinterbase = pe.Constraint(model.ind,
                                              rule=__bounds_inter_base)
    model.con_boundsinterpeak = pe.Constraint(model.ind,
                                              rule=__bounds_inter_peak)


###############################################################################
# noinspection PyProtectedMember
def __base_losses(mod, ii):
    base = mod._base
    efficiency_losses = (mod.baseplus[ii] * base.efficiency.charge +
                         mod.baseminus[ii] / base.efficiency.discharge)
    if ii is 0:
        discharge_losses = -mod.baseenergyinit/base.selfdischarge
    else:
        discharge_losses = -mod.baseenergy[ii-1]/base.selfdischarge
    return mod.baseinner[ii] == efficiency_losses + discharge_losses


# noinspection PyProtectedMember
def __peak_losses(mod, ii):
    peak = mod._peak
    efficiency_losses = (mod.peakplus[ii] * peak.efficiency.charge +
                         mod.peakminus[ii] / peak.efficiency.discharge)
    if ii is 0:
        discharge_losses = -mod.peakenergyinit/peak.selfdischarge
    else:
        discharge_losses = -mod.peakenergy[ii - 1]/peak.selfdischarge
    return mod.peakinner[ii] == efficiency_losses + discharge_losses


def _loss_model_constraint(model):
    """Implements efficiency losses and self discharge losses and links base
    power with inner base power (and peak power respectively)"""
    model.con_baselosses = pe.Constraint(model.ind, rule=__base_losses)
    model.con_peaklosses = pe.Constraint(model.ind, rule=__peak_losses)


###############################################################################
# constraint integrate energy - connect power and energy
def __integrate_base(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    if ii is 0:
        lastenergy = mod.baseenergyinit
    else:
        lastenergy = mod.baseenergy[ii - 1]
    return (mod.baseenergy[ii] == lastenergy +
            (mod.inter[ii] + mod.baseinner[ii]) * dtimes[ii])


def __integrate_peak(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    if ii is 0:
        lastenergy = mod.peakenergyinit
    else:
        lastenergy = mod.peakenergy[ii - 1]
    return (mod.peakenergy[ii] == lastenergy +
            (-mod.inter[ii] + mod.peakinner[ii]) * dtimes[ii])


def _integrate_power_constraint(model):
    """Euler Integration per timestep of power to gain energy"""
    model.con_integratebase = pe.Constraint(model.ind, rule=__integrate_base)
    model.con_integratepeak = pe.Constraint(model.ind, rule=__integrate_peak)


###############################################################################
def _cyclic_energy_constraint(model):
    """Ensures that energy content at beginning of interval equals energy
    content at end of interval"""

    model.con_cyclicbase = \
        pe.Constraint(expr=(model.baseenergyinit ==
                            model.baseenergy[model.ind.last()]))
    model.con_cyclicpeak = \
        pe.Constraint(expr=(model.peakenergyinit ==
                            model.peakenergy[model.ind.last()]))


###############################################################################
def __energy_base_lower_max(mod, ii):
    return mod.baseenergy[ii] <= mod.baseenergycapacity


def __energy_peak_lower_max(mod, ii):
    return mod.peakenergy[ii] <= mod.peakenergycapacity


def _energy_lower_max_constraint(model):
    """Ensures that energy capacity of storages is not exceeded for all
    times"""
    model.con_baseenergylowermax = pe.Constraint(model.ind,
                                                 rule=__energy_base_lower_max)
    model.con_peakenergylowermax = pe.Constraint(model.ind,
                                                 rule=__energy_peak_lower_max)

    model.con_baseenergyinitlowermax = \
        pe.Constraint(expr=(model.baseenergyinit <= model.baseenergycapacity))
    model.con_peakenergyinitlowermax = \
        pe.Constraint(expr=(model.peakenergyinit <= model.peakenergycapacity))


###############################################################################
# Base binary bounds
def __bin_bound_base_lower(mod, ii):
    # noinspection PyProtectedMember
    base = mod._base
    return base.power.min*mod.binbaselower[ii] <= mod.base[ii]


def __bin_bound_base_upper(mod, ii):
    # noinspection PyProtectedMember
    base = mod._base
    return mod.base[ii] <= base.power.max*mod.binbaseupper[ii]


def __bin_bound_baseminus_lower(mod, ii):
    # noinspection PyProtectedMember
    base = mod._base
    return base.power.min*mod.binbaselower[ii] <= mod.baseminus[ii]


def __bin_bound_baseplus_upper(mod, ii):
    # noinspection PyProtectedMember
    base = mod._base
    return mod.baseplus[ii] <= base.power.max*mod.binbaseupper[ii]


# Peak binary bounds
def __bin_bound_peak_lower(mod, ii):
    # noinspection PyProtectedMember
    peak = mod._peak
    return peak.power.min*mod.binpeaklower[ii] <= mod.peak[ii]


def __bin_bound_peak_upper(mod, ii):
    # noinspection PyProtectedMember
    peak = mod._peak
    return mod.peak[ii] <= peak.power.max*mod.binpeakupper[ii]


def __bin_bound_peakminus_lower(mod, ii):
    # noinspection PyProtectedMember
    peak = mod._peak
    return peak.power.min*mod.binpeaklower[ii] <= mod.peakminus[ii]


def __bin_bound_peakplus_upper(mod, ii):
    # noinspection PyProtectedMember
    peak = mod._peak
    return mod.peakplus[ii] <= peak.power.max*mod.binpeakupper[ii]


# Inter Power Flow Binary Bounds; firstly, define (continuous) bounds
# noinspection PyProtectedMember
def __bin_bound_inter_lower(mod, ii):
    peak = mod._peak
    base = mod._base
    inter_minpower = max(base.power.min/base.efficiency.discharge,
                         peak.power.min/peak.efficiency.discharge)
    return inter_minpower*mod.bininterlower[ii] <= mod.inter[ii]


# noinspection PyProtectedMember
def __bin_bound_inter_upper(mod, ii):
    peak = mod._peak
    base = mod._base
    inter_maxpower = min(base.power.max*base.efficiency.charge,
                         peak.power.max*peak.efficiency.charge)
    return mod.inter[ii] <= inter_maxpower*mod.bininterupper[ii]


def _binary_bounds_constraint(model):
    """Buts a binary variable on all relevant power bounds"""

    # Base binary bounds
    model.con_bin_bound_base_lower = \
        pe.Constraint(model.ind, rule=__bin_bound_base_lower)
    model.con_bin_bound_base_upper = \
        pe.Constraint(model.ind, rule=__bin_bound_base_upper)

    model.con_bin_bound_baseminus_lower = \
        pe.Constraint(model.ind, rule=__bin_bound_baseminus_lower)
    model.con_bin_bound_baseplus_upper = \
        pe.Constraint(model.ind, rule=__bin_bound_baseplus_upper)

    # Peak binary bounds
    model.con_bin_bound_peak_lower = \
        pe.Constraint(model.ind, rule=__bin_bound_peak_lower)
    model.con_bin_bound_peak_upper = \
        pe.Constraint(model.ind, rule=__bin_bound_peak_upper)

    model.con_bin_bound_peakminus_lower = \
        pe.Constraint(model.ind, rule=__bin_bound_peakminus_lower)
    model.con_bin_bound_peakplus_upper = \
        pe.Constraint(model.ind, rule=__bin_bound_peakplus_upper)

    # Inter Power Flow Binary Bounds
    model.con_bin_bound_inter_lower = \
        pe.Constraint(model.ind, rule=__bin_bound_inter_lower)
    model.con_bin_bound_inter_upper = \
        pe.Constraint(model.ind, rule=__bin_bound_inter_upper)


###############################################################################
def __bin_base_interval(mod, ii):
    """Ensures that base bounds are opened to either positive real axis
    or negative real axes"""
    return mod.binbaselower[ii] + mod.binbaseupper[ii] <= 1


def __bin_peak_interval(mod, ii):
    """Ensures that peak bounds are opened to either positive real axis
    or negative real axes"""
    return mod.binpeaklower[ii] + mod.binpeakupper[ii] <= 1


def __bin_inter_interval(mod, ii):
    """Ensures that inter bounds are opened to either positive real axis
    or negative real axes"""
    return mod.bininterlower[ii] + mod.bininterupper[ii] <= 1


def _binary_interval_locks(model):
    """Connects binary variables to ensure that only one direction of real
    axis for base, peak and inter is opened"""
    model.con_bin_base_interval = \
        pe.Constraint(model.ind, rule=__bin_base_interval)
    model.con_bin_peak_interval = \
        pe.Constraint(model.ind, rule=__bin_peak_interval)
    model.con_bin_inter_interval = \
        pe.Constraint(model.ind, rule=__bin_inter_interval)


###############################################################################
def __cross_base_peak(mod, ii):
    return mod.binbaseupper[ii] + mod.binpeaklower[ii] <= 1


def __cross_peak_base(mod, ii):
    return mod.binpeakupper[ii] + mod.binbaselower[ii] <= 1


def __cross_base_inter(mod, ii):
    return mod.binbaseupper[ii] + mod.bininterlower[ii] <= 1


def __cross_inter_base(mod, ii):
    return mod.bininterupper[ii] + mod.binbaselower[ii] <= 1


def __cross_peak_inter(mod, ii):
    return mod.binpeakupper[ii] + mod.bininterupper[ii] <= 1


def __cross_inter_peak(mod, ii):
    return mod.bininterlower[ii] + mod.binpeaklower[ii] <= 1


def _binary_cross_locks(model):
    """Connects left peak with right right base and so on"""
    model.con_cross_base_peak = \
        pe.Constraint(model.ind, rule=__cross_base_peak)
    model.con_cross_peak_base = \
        pe.Constraint(model.ind, rule=__cross_peak_base)

    model.con_cross_base_inter = \
        pe.Constraint(model.ind, rule=__cross_base_inter)
    model.con_cross_inter_base = \
        pe.Constraint(model.ind, rule=__cross_inter_base)

    model.con_cross_peak_inter = \
        pe.Constraint(model.ind, rule=__cross_peak_inter)
    model.con_cross_inter_peak = \
        pe.Constraint(model.ind, rule=__cross_inter_peak)


###############################################################################
def __number_upper(mod, ii):
    return (mod.binbaseupper[ii] + mod.binpeakupper[ii] +
            mod.bininterupper[ii] <= 2)


def __number_lower(mod, ii):
    return (mod.binbaselower[ii] + mod.binpeaklower[ii] +
            mod.bininterlower[ii] <= 2)


def _binary_number_locks(model):
    """Defines that a maximum of 2 in (base, peak, inter) can be nonzero"""
    # Redundant and implicitely fulfilled with other binary constraints
    model.con_bin_number_upper = pe.Constraint(model.ind, rule=__number_upper)
    model.con_bin_number_lower = pe.Constraint(model.ind, rule=__number_lower)
