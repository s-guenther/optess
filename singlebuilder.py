#!/usr/bin/env python3
"""This module aggregates helper functions related to module 'optimmodel'
Its related to building the pyomo model for HESS"""

import pyomo.environ as pe
from copy import deepcopy
from powersignal import Signal
from objective import Objective
from storage import Storage


class SingleBuilder:
    def __init__(self):
        self.signal = None
        self.storage = None
        self.objective = None
        self.model = None
        self.model_2nd = None

    # noinspection PyArgumentList
    def build_1st_stage(self, signal, storage, objective,
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

    def build_2nd_stage(self, name='Single ESS Model, '
                                   '2nd Quadratic Minimizing Stage'):
        self.model_2nd = deepcopy(self.model)
        model = self.model_2nd

        model.name = name

        # Fix previously determined energy capacity
        storagedim = list(model.energycapacity.get_values().values())[0]
        self.model_2nd.con_lock_energy_capacity = \
            pe.Constraint(expr=model.energycapacity == storagedim)

        # Add quadratic minimizing expression
        model.objexpr = sum(model.powerplus[ii]**2 + model.powerminus[ii]**2
                            for ii in model.ind)
        model.del_component(model.obj)
        model.obj = pe.Objective(expr=model.objexpr)

        return model

    def build(self, signal, storage, objective, name='Single ESS Model'):
        return self.build_1st_stage(signal, storage, objective, name)

    def _add_vars(self):
        """Defines all Variables of the optimization model"""
        # Extract Variables from Data
        signal = self.signal
        storage = self.storage
        model = self.model

        # Set for range of variables
        model.ind = pe.Set(initialize=range(len(signal)), ordered=True)

        # Create empty objective expression, which is filled in subsequent
        # steps
        model.objexpr = 0

        _add_power_vars(model, storage)
        _add_energy_vars(model)

    def _add_constraints(self):
        """Defines all Constraints of the optimization model"""
        # Extract Variables from data
        signal = self.signal
        storage = self.storage
        model = self.model

        _lock_plus_and_minus_constraint(model)
        _loss_model_constraint(model, signal, storage)
        _integrate_power_constraint(model, signal)
        _cyclic_energy_constraint(model)
        _energy_lower_max_constraint(model)

    def _add_capacity_minimizing_objective(self):
        """Adds objective function that minimizes energy capacity of peak and
        base. Must be called last."""
        model = self.model

        model.objexpr += model.energycapacity
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
            return signal[ii] - mod.power[ii] >= minpower

        def cutting_high(mod, ii):
            return signal[ii] - mod.power[ii] <= maxpower

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
            return (signal[ii] - mod.power[ii] <=
                    mod.deltaplus[ii] + mod.deltaminus[ii])

        model.con_split_delta = pe.Constraint(model.ind, rule=split_delta)

        model.con_throughput = \
            pe.Constraint(expr=sum(model.deltaplus[ii]*dtime[ii] <= maxenergy
                                   for ii in model.ind))


# ###
# ### Elementary Functions
# ###

def _add_power_vars(model, storage):
    # Base power and peak power
    model.power = pe.Var(model.ind, bounds=(storage.power.min,
                                            storage.power.max))

    # Positive and negative part of base and peak power
    model.powerplus = pe.Var(model.ind, bounds=(0, storage.power.max))
    model.powerminus = pe.Var(model.ind, bounds=(storage.power.min, 0))

    # Ideal inner base and peak power considering losses
    model.inner = pe.Var(model.ind)


def _add_energy_vars(model):
    # Energy content as function of time
    model.energy = pe.Var(model.ind, bounds=(0, None))

    # Energy capacity (maximum storable energy)
    model.energycapacity = pe.Var(bounds=(0, None))

    # Initial condition of base and peak energy content
    model.energyinit = pe.Var(bounds=(0, None))


def _lock_plus_and_minus_constraint(model, multiplier=1):
    """ensure that powerplus + powerminus = power, this also adds a soft
    constraint to the objective expression, minimizing powerplus - powerminus
    to prevent something like   power = plus + minus <==> 5 = 7 + -2
    and gain                    power = plus + minus <==> 5 = 5 + 0 instead"""

    def lock_power(mod, ii):
        return mod.power[ii] == mod.powerplus[ii] + mod.powerminus[ii]

    model.con_lockpower = pe.Constraint(model.ind, rule=lock_power)

    model.objexpr += sum(model.powerplus[ii] - model.powerminus[ii]
                         for ii in model.ind)*multiplier


def _loss_model_constraint(model, signal, storage):
    """Implements efficiency losses and self discharge losses and links base
    power with inner base power (and peak power respectively)"""

    dtimes = signal.dtimes

    def losses(mod, ii):
        efficiency_losses = (mod.powerplus[ii] *
                             storage.efficiency.charge +
                             mod.powerminus[ii] /
                             storage.efficiency.discharge)
        if ii is 0:
            discharge_losses = -mod.energyinit * \
                               storage.selfdischarge*dtimes[ii]
        else:
            discharge_losses = -mod.energy[ii - 1] * \
                               storage.selfdischarge*dtimes[ii]
        return mod.inner[ii] == efficiency_losses + discharge_losses

    model.con_storagelosses = pe.Constraint(model.ind, rule=losses)


def _integrate_power_constraint(model, signal):
    """Euler Integration per timestep of power to gain energy"""

    dtimes = signal.dtimes

    # constraint integrate energy - connect power and energy
    def integrate(mod, ii):
        if ii is 0:
            lastenergy = mod.energyinit
        else:
            lastenergy = mod.energy[ii - 1]
        return mod.energy[ii] == lastenergy + mod.inner[ii] * dtimes[ii]

    model.con_integrate = pe.Constraint(model.ind, rule=integrate)


def _cyclic_energy_constraint(model):
    """Ensures that energy content at beginning of interval equals energy
    content at end of interval"""

    model.con_cyclic = pe.Constraint(expr=(model.energyinit ==
                                           model.energy[model.ind.last()]))


def _energy_lower_max_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""

    def energy_lower_max(mod, ii):
        return mod.energy[ii] <= mod.energycapacity

    model.con_energylowermax = pe.Constraint(model.ind,
                                             rule=energy_lower_max)

    model.con_energyinitlowermax = \
        pe.Constraint(expr=(model.energyinit <= model.energycapacity))
