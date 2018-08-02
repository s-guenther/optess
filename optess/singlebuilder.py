#!/usr/bin/env python3
"""This module aggregates helper functions related to module 'optimmodel'
Its related to building the pyomo model for HESS"""

import pyomo.environ as pe
from copy import deepcopy

from .signal import Signal
from .objective import Objective
from .storage import Storage


class SingleBuilder:
    def __init__(self):
        self.signal = None
        self.storage = None
        self.objective = None
        self.model = None
        self.model_2nd = None

    ###########################################################################
    # noinspection PyArgumentList
    def minimize_energy(self, signal, storage, objective,
                        name='Single ESS Model, 1st Dimensioning Stage'):
        self.signal = Signal(signal)
        self.storage = Storage(storage)
        self.objective = Objective(objective)

        self.model = pe.ConcreteModel(name)

        self._add_vars()
        self._add_constraints()

        if objective.type.name == 'power':
            self._add_peak_cutting_objective()
        else:
            self._add_throughput_objective()

        self._add_capacity_minimizing_objective()

        return self.model

    ###########################################################################
    def minimize_cycles(self, name='Single ESS Model, '
                                   '2nd Quadratic Minimizing Stage'):
        self.model_2nd = deepcopy(self.model)
        model = self.model_2nd

        model.name = name

        # Fix previously determined energy capacity
        storagedim = list(model.energycapacity.get_values().values())[0]
        self.model_2nd.con_lock_energy_capacity = \
            pe.Constraint(expr=model.energycapacity == storagedim)

        # Add quadratic minimizing expression
        model.objexpr = sum(model.powerplus[ii] + model.powerminus[ii]
                            for ii in model.ind)
        model.del_component(model.obj)
        model.obj = pe.Objective(expr=model.objexpr)

        return model

    ###########################################################################
    def build(self, signal, storage, objective, name='Single ESS Model'):
        return self.minimize_energy(signal, storage, objective, name)

    ###########################################################################
    def _add_vars(self):
        """Defines all Variables of the optimization model"""
        # Extract Variables from Data
        signal = self.signal
        storage = self.storage
        objective = self.objective
        model = self.model

        # Set for range of variables
        model.ind = pe.Set(initialize=range(len(signal)), ordered=True)

        # Create empty objective expression, which is filled in subsequent
        # steps
        model.objexpr = 0

        _add_aux_vars(model, signal, storage, objective)
        _add_power_vars(model, storage)
        _add_energy_vars(model)

    ###########################################################################
    def _add_constraints(self):
        """Defines all Constraints of the optimization model"""
        # Extract Variables from data
        model = self.model

        _lock_plus_and_minus_constraint(model)
        _loss_model_constraint(model)
        _integrate_power_constraint(model)
        _cyclic_energy_constraint(model)
        _energy_lower_max_constraint(model)

    ###########################################################################
    def _add_capacity_minimizing_objective(self):
        """Adds objective function that minimizes energy capacity of peak and
        base. Must be called last."""
        model = self.model

        model.objexpr += model.energycapacity
        model.obj = pe.Objective(expr=model.objexpr)

    ###########################################################################
    def _add_peak_cutting_objective(self):
        """Add objective - this is an objective or aim in a larger sense as it
        will cut peak power which also adds constraints to reach the
        objective."""
        model = self.model

        model.con_cutting_low = \
            pe.Constraint(model.ind, rule=_cutting_low)
        model.con_cutting_high = \
            pe.Constraint(model.ind, rule=_cutting_high)

    ###########################################################################
    def _add_throughput_objective(self):
        """Add objective - this is an objective or aim in a larger sense as it
        will decrease the amount of energy taken from grid/supply etc. It will
        also add constraints to reach objective. This objective only makes
        sense if the power from grid changes sign, i.e. if power is fed into
        grid"""
        dtimes = self.signal.dtimes
        maxenergy = self.objective.val.max  # TODO, formulation makes no sense
        model = self.model

        model.deltaplus = pe.Var(model.ind, bounds=(0, None))
        model.deltaminus = pe.Var(model.ind, bounds=(None, 0))

        model.con_split_delta = \
            pe.Constraint(model.ind, rule=_split_delta)

        model.con_throughput = \
            pe.Constraint(expr=sum(model.deltaplus[ii]*dtimes[ii]
                                   for ii in model.ind) <= maxenergy )


###########################################################################
# _add_peak_cutting_objective(self):
# noinspection PyProtectedMember
def _cutting_low(mod, ii):
    signal = mod._signal.vals
    minpower = mod._objective.val.min
    return signal[ii] + mod.power[ii] >= minpower


# noinspection PyProtectedMember
def _cutting_high(mod, ii):
    signal = mod._signal.vals
    maxpower = mod._objective.val.max
    return signal[ii] + mod.power[ii] <= maxpower


###########################################################################
# _add_throughput_objective(self):
def _split_delta(mod, ii):
    # noinspection PyProtectedMember
    signal = mod._signal.vals
    return (signal[ii] + mod.power[ii] <=
            mod.deltaplus[ii] + mod.deltaminus[ii])


# ###
# ### Elementary Functions
# ###

###############################################################################
def _add_aux_vars(model, signal, storage, objective):
    model._signal = signal
    model._storage = storage
    model._objective = objective


###############################################################################
def _add_power_vars(model, storage):
    # Base power and peak power
    model.power = pe.Var(model.ind, bounds=(storage.power.min,
                                            storage.power.max))

    # Positive and negative part of base and peak power
    model.powerplus = pe.Var(model.ind, bounds=(0, storage.power.max))
    model.powerminus = pe.Var(model.ind, bounds=(storage.power.min, 0))

    # Ideal inner base and peak power considering losses
    model.inner = pe.Var(model.ind)


###############################################################################
def _add_energy_vars(model):
    # Energy content as function of time
    model.energy = pe.Var(model.ind, bounds=(0, None))

    # Energy capacity (maximum storable energy)
    model.energycapacity = pe.Var(bounds=(0, None))

    # Initial condition of base and peak energy content
    model.energyinit = pe.Var(bounds=(0, None))


###############################################################################
def __lock_power(mod, ii):
    return mod.power[ii] == mod.powerplus[ii] + mod.powerminus[ii]


def _lock_plus_and_minus_constraint(model, multiplier=1):
    """ensure that powerplus + powerminus = power, this also adds a soft
    constraint to the objective expression, minimizing powerplus - powerminus
    to prevent something like   power = plus + minus <==> 5 = 7 + -2
    and gain                    power = plus + minus <==> 5 = 5 + 0 instead"""

    model.con_lockpower = pe.Constraint(model.ind, rule=__lock_power)

    model.objexpr += sum(model.powerplus[ii] - model.powerminus[ii]
                         for ii in model.ind)*multiplier


###############################################################################
# noinspection PyProtectedMember
def __losses(mod, ii):
    storage = mod._storage
    dtimes = mod._signal.dtimes
    efficiency_losses = (mod.powerplus[ii]*storage.efficiency.charge +
                         mod.powerminus[ii]/storage.efficiency.discharge)
    if ii is 0:
        discharge_losses = -mod.energyinit/storage.selfdischarge
    else:
        discharge_losses = -mod.energy[ii-1]/storage.selfdischarge
    return mod.inner[ii] == efficiency_losses + discharge_losses


def _loss_model_constraint(model):
    """Implements efficiency losses and self discharge losses and links base
    power with inner base power (and peak power respectively)"""
    model.con_storagelosses = pe.Constraint(model.ind, rule=__losses)


###############################################################################
# constraint integrate energy - connect power and energy
def __integrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    if ii is 0:
        lastenergy = mod.energyinit
    else:
        lastenergy = mod.energy[ii - 1]
    return mod.energy[ii] == lastenergy + mod.inner[ii] * dtimes[ii]


def _integrate_power_constraint(model):
    """Euler Integration per timestep of power to gain energy"""
    model.con_integrate = pe.Constraint(model.ind, rule=__integrate)


###############################################################################
def _cyclic_energy_constraint(model):
    """Ensures that energy content at beginning of interval equals energy
    content at end of interval"""

    model.con_cyclic = pe.Constraint(expr=(model.energyinit ==
                                           model.energy[model.ind.last()]))


###############################################################################
def __energy_lower_max(mod, ii):
    return mod.energy[ii] <= mod.energycapacity


def _energy_lower_max_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""
    model.con_energylowermax = pe.Constraint(model.ind,
                                             rule=__energy_lower_max)

    model.con_energyinitlowermax = \
        pe.Constraint(expr=(model.energyinit <= model.energycapacity))
