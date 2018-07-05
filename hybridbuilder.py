#!/usr/bin/env python3
"""This module aggregates helper functions related to module 'optimmodel'
Its related to building the pyomo model for HESS"""

import pyomo.environ as pe
from copy import deepcopy

from objective import Objective, Strategy
from powersignal import Signal
from storage import Storage


class HybridBuilder:
    def __init__(self):
        self.signal = None
        self.base = None
        self.peak = None
        self.objective = None
        self.strategy = None
        self.model = None
        self.model_2nd = None

    # noinspection PyArgumentList
    def build_1st_stage(self, signal, base, peak, objective,
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

    def build_2nd_stage(self, multiplier=1e-1):
        self.model_2nd = deepcopy(self.model)
        model = self.model_2nd

        # Fix previously determined base and peak energy capacity
        basedim = list(model.baseenergycapacity.get_values().values())[0]
        peakdim = list(model.peakenergycapacity.get_values().values())[0]
        self.model_2nd.con_lock_baseenergy_capacity = \
            pe.Constraint(expr=model.baseenergycapacity == basedim)
        self.model_2nd.con_lock_peakenergy_capacity = \
            pe.Constraint(expr=model.peakenergycapacity == peakdim)

        # Add quadratic minimizing expression
        model.interplus = pe.Var(model.ind, bounds=(0, None))
        model.interminus = pe.Var(model.ind, bounds=(None, 0))

        def lock_inter(mod, ii):
            return mod.inter[ii] == mod.interplus[ii] + mod.interminus[ii]

        model.con_lockinter = pe.Constraint(model.ind, rule=lock_inter)

        # Define a monotonically incrasing vector which will be multiplied
        # with interminus. This way, interstorage power flow becomes
        # determined and unambiguous (hopefully)
        monodec = [(3*model.ind.last() - ii)/3 for ii in model.ind]

        model.objexpr = sum((-model.baseminus[ii]
                             + (model.interplus[ii] -
                                model.interminus[ii])
                             - model.peakminus[ii]*multiplier)*monodec[ii]
                            for ii in model.ind)
        model.del_component(model.obj)
        model.obj = pe.Objective(expr=model.objexpr)

        return model

    def build(self, signal, base, peak, objective, strategy, name):
        return self.build_1st_stage(signal, base, peak,
                                    objective, strategy, name)

    def _add_vars(self):
        """Defines all Variables of the optimization model"""
        # Extract Variables from Data
        signal = self.signal
        base = self.base
        peak = self.peak
        model = self.model

        # Set for range of variables
        self.model.ind = pe.Set(initialize=range(len(signal)), ordered=True)

        # Create empty objective expression, which is filled in subsequent
        # steps
        self.model.objexpr = 0

        _add_power_vars(model, base, peak)
        _add_energy_vars(model)
        _add_binary_vars(model)

    def _add_constraints(self):
        """Defines all Constraints of the optimization model"""
        # Extract Variables from data
        signal = self.signal
        base = self.base
        peak = self.peak
        model = self.model

        _lock_plus_and_minus_constraint(model)
        _bounds_for_inter_constraint(model, base, peak)
        _loss_model_constraint(model, signal, base, peak)
        _integrate_power_constraint(model, signal)
        _cyclic_energy_constraint(model)
        _energy_lower_max_constraint(model)
        _binary_bounds_constraint(model, base, peak)
        _binary_interval_locks(model)
        _binary_cross_locks(model)
        # _binary_number_locks(model)

    def _add_inter_constraint(self):
        """Adds constraint that allows an inter-storage power flow. This
        function does nothing as it is automatically satisfied, but is included
        for symmetry"""
        pass

    def _add_nointer_constraint(self):
        """Sets inter storage power flow to zero to prevent reloading"""
        model = self.model

        def nointer_constraint(mod, ii):
            return mod.inter[ii] == 0

        def bin_nointer_constraint(mod, ii):
            return mod.bininterlower[ii] + mod.bininterupper[ii] == 0

        model.con_nointer = pe.Constraint(model.ind, rule=nointer_constraint)
        model.con_bin_nointer = pe.Constraint(model.ind,
                                              rule=bin_nointer_constraint)

    def _add_capacity_minimizing_objective(self, multiplier=0.50):
        """Adds objective function that minimizes energy capacity of peak and
        base, add this objective last"""
        model = self.model
        model.objexpr += multiplier*model.baseenergycapacity
        model.objexpr += model.peakenergycapacity
        model.obj = pe.Objective(expr=model.objexpr)

    def _add_peak_cutting_objective(self):
        """Add objective - this is an objective or aim in a larger sense as it
        will cut peak power which also adds constraints to reach the
        objective."""
        signal = self.signal.vals
        minpower = self.objective.val.min
        maxpower = self.objective.val.max
        model = self.model

        def cutting_low(mod, ii):
            return signal[ii] - (mod.base[ii] + mod.peak[ii]) >= minpower

        def cutting_high(mod, ii):
            return signal[ii] - (mod.base[ii] + mod.peak[ii]) <= maxpower

        model.con_cutting_low = pe.Constraint(model.ind, rule=cutting_low)
        model.con_cutting_high = pe.Constraint(model.ind, rule=cutting_high)

    def _add_throughput_objective(self):
        """Add objective - this is an objective or aim in a larger sense as it
        will decrease the amount of energy taken from grid/supply etc. It will
        also add constraints to reach objective. This objective only makes
        sense if the power from grid changes sign, i.e. if power is fed into
        grid"""
        signal = self.signal.vals
        dtime = self.signal.dtimes
        maxenergy = self.objective.val
        model = self.model

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
                         bounds=(max(base.power.min, -peak.power.max),
                                 min(base.power.max, -peak.power.min)))


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
    model.binbaselower = pe.Var(model.ind, within=pe.Binary)
    model.binbaseupper = pe.Var(model.ind, within=pe.Binary)

    # Binary variable for peak lower and upper bound switch
    model.binpeaklower = pe.Var(model.ind, within=pe.Binary)
    model.binpeakupper = pe.Var(model.ind, within=pe.Binary)

    # Binary variable for inter storage power lower and upper bound switch
    model.bininterlower = pe.Var(model.ind, within=pe.Binary)
    model.bininterupper = pe.Var(model.ind, within=pe.Binary)


def _lock_plus_and_minus_constraint(model, multiplier=1):
    """ensure that powerplus + powerminus = power, this also adds a penalty
    term to the objective expression, minimizing powerplus - powerminus
    to prevent something like   power = plus + minus <==> 5 = 7 + -2
    and gain                    power = plus + minus <==> 5 = 5 + 0 instead"""

    def lock_base(mod, ii):
        return mod.base[ii] == mod.baseplus[ii] + mod.baseminus[ii]

    def lock_peak(mod, ii):
        return mod.peak[ii] == mod.peakplus[ii] + mod.peakminus[ii]

    model.con_lockbase = pe.Constraint(model.ind, rule=lock_base)
    model.con_lockpeak = pe.Constraint(model.ind, rule=lock_peak)

    model.objexpr += sum(model.baseplus[ii] - model.baseminus[ii] +
                         model.peakplus[ii] - model.peakminus[ii]
                         for ii in model.ind)*multiplier


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

    dtimes = signal.dtimes

    def base_losses(mod, ii):
        efficiency_losses = (mod.baseplus[ii] * base.efficiency.charge +
                             mod.baseminus[ii] / base.efficiency.discharge)
        if ii is 0:
            discharge_losses = -mod.baseenergyinit * \
                               base.selfdischarge*dtimes[ii]
        else:
            discharge_losses = -mod.baseenergy[ii - 1] * \
                               base.selfdischarge*dtimes[ii]
        return mod.baseinner[ii] == efficiency_losses + discharge_losses

    def peak_losses(mod, ii):
        efficiency_losses = (mod.peakplus[ii] * peak.efficiency.charge +
                             mod.peakminus[ii] / peak.efficiency.discharge)
        if ii is 0:
            discharge_losses = -mod.peakenergyinit * \
                               peak.selfdischarge*dtimes[ii]
        else:
            discharge_losses = -mod.peakenergy[ii - 1] * \
                               peak.selfdischarge*dtimes[ii]
        return mod.peakinner[ii] == efficiency_losses + discharge_losses

    model.con_baselosses = pe.Constraint(model.ind, rule=base_losses)
    model.con_peaklosses = pe.Constraint(model.ind, rule=peak_losses)


def _integrate_power_constraint(model, signal):
    """Euler Integration per timestep of power to gain energy"""

    dtimes = signal.dtimes

    # constraint integrate energy - connect power and energy
    def integrate_base(mod, ii):
        if ii is 0:
            lastenergy = mod.baseenergyinit
        else:
            lastenergy = mod.baseenergy[ii - 1]
        return (mod.baseenergy[ii] == lastenergy +
                (mod.inter[ii] + mod.baseinner[ii]) * dtimes[ii])

    def integrate_peak(mod, ii):
        if ii is 0:
            lastenergy = mod.peakenergyinit
        else:
            lastenergy = mod.peakenergy[ii - 1]
        return (mod.peakenergy[ii] == lastenergy +
                (-mod.inter[ii] + mod.peakinner[ii]) * dtimes[ii])

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
    """Ensures that energy capacity of storages is not exceeded for all
    times"""
    def energy_base_lower_max(mod, ii):
        return mod.baseenergy[ii] <= mod.baseenergycapacity

    def energy_peak_lower_max(mod, ii):
        return mod.peakenergy[ii] <= mod.peakenergycapacity

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

    # Base binary bounds
    def bin_bound_base_lower(mod, ii):
        return base.power.min*mod.binbaselower[ii] <= mod.base[ii]

    def bin_bound_base_upper(mod, ii):
        return mod.base[ii] <= base.power.max*mod.binbaseupper[ii]

    def bin_bound_baseminus_lower(mod, ii):
        return base.power.min*mod.binbaselower[ii] <= mod.baseminus[ii]

    def bin_bound_baseplus_upper(mod, ii):
        return mod.baseplus[ii] <= base.power.max*mod.binbaseupper[ii]

    model.con_bin_bound_base_lower = \
        pe.Constraint(model.ind, rule=bin_bound_base_lower)
    model.con_bin_bound_base_upper = \
        pe.Constraint(model.ind, rule=bin_bound_base_upper)

    model.con_bin_bound_baseminus_lower = \
        pe.Constraint(model.ind, rule=bin_bound_baseminus_lower)
    model.con_bin_bound_baseplus_upper = \
        pe.Constraint(model.ind, rule=bin_bound_baseplus_upper)

    # Peak binary bounds
    def bin_bound_peak_lower(mod, ii):
        return peak.power.min*mod.binpeaklower[ii] <= mod.peak[ii]

    def bin_bound_peak_upper(mod, ii):
        return mod.peak[ii] <= peak.power.max*mod.binpeakupper[ii]

    def bin_bound_peakminus_lower(mod, ii):
        return peak.power.min*mod.binpeaklower[ii] <= mod.peakminus[ii]

    def bin_bound_peakplus_upper(mod, ii):
        return mod.peakplus[ii] <= peak.power.max*mod.binpeakupper[ii]

    model.con_bin_bound_peak_lower = \
        pe.Constraint(model.ind, rule=bin_bound_peak_lower)
    model.con_bin_bound_peak_upper = \
        pe.Constraint(model.ind, rule=bin_bound_peak_upper)

    model.con_bin_bound_peakminus_lower = \
        pe.Constraint(model.ind, rule=bin_bound_peakminus_lower)
    model.con_bin_bound_peakplus_upper = \
        pe.Constraint(model.ind, rule=bin_bound_peakplus_upper)

    # Inter Power Flow Binary Bounds; firstly, define (continuous) bounds
    inter_minpower = max(base.power.min/base.efficiency.discharge,
                         peak.power.min/peak.efficiency.discharge)
    inter_maxpower = min(base.power.max*base.efficiency.charge,
                         peak.power.max*peak.efficiency.charge)

    def bin_bound_inter_lower(mod, ii):
        return inter_minpower*mod.bininterlower[ii] <= mod.inter[ii]

    def bin_bound_inter_upper(mod, ii):
        return mod.inter[ii] <= inter_maxpower*mod.bininterupper[ii]

    model.con_bin_bound_inter_lower = pe.Constraint(model.ind,
                                                    rule=bin_bound_inter_lower)
    model.con_bin_bound_inter_upper = pe.Constraint(model.ind,
                                                    rule=bin_bound_inter_upper)


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
        return mod.binpeakupper[ii] + mod.bininterupper[ii] <= 1

    def cross_inter_peak(mod, ii):
        return mod.bininterlower[ii] + mod.binpeaklower[ii] <= 1

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


def _binary_number_locks(model):
    """Defines that a maximum of 2 in (base, peak, inter) can be nonzero"""
    # Redundant and implicitely fulfilled with other binary constraints
    def number_upper(mod, ii):
        return (mod.binbaseupper[ii] + mod.binpeakupper[ii] +
                mod.bininterupper[ii] <= 2)

    def number_lower(mod, ii):
        return (mod.binbaselower[ii] + mod.binpeaklower[ii] +
                mod.bininterlower[ii] <= 2)

    model.con_bin_number_upper = pe.Constraint(model.ind, rule=number_upper)
    model.con_bin_number_lower = pe.Constraint(model.ind, rule=number_lower)
