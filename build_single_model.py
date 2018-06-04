#!/usr/bin/env python3
"""This module aggregates helper functions related to module 'optimmodel'
Its related to building the pyomo model for HESS"""

import pyomo.environ as pe


# ###
# ### Wrapper functions
# ###

def add_vars(data, model):
    """Defines all Variables of the optimization model"""
    # Extract Variables from Data
    signal = data.signal
    storage = data.storage

    # Set for range of variables
    model.ind = pe.Set(initialize=range(len(signal)), ordered=True)

    # Create empty objective expression, which is filled in subsequent steps
    model.objexpr = 0

    _add_power_vars(model, storage)
    _add_energy_vars(model)


def add_constraints(data, model):
    """Defines all Constraints of the optimization model"""
    # Extract Variables from data
    signal = data.signal
    storage = data.storage

    # TODO
    _lock_plus_and_minus_constraint(model)
    _loss_model_constraint(model, signal, storage)
    _integrate_power_constraint(model, signal)
    _cyclic_energy_constraint(model)


def add_capacity_minimizing_objective(model, multiplier=0.99):
    """Adds objective function that minimizes energy capacity of peak and
    base"""
    model.objexpr += model.peakenergycapacity
    model.obj = pe.Objective(expr=model.objexpr)


def add_peak_cutting_objective(data, model):
    """Add objective - this is an objective or aim in a larger sense as it
    will cut peak power which also adds constraints to reach the objective."""
    signal = data.signal.value
    minpower = data.objective.value.min
    maxpower = data.objective.value.max

    def cutting_low(mod, ii):
        return signal[ii] - mod.power[ii] >= minpower

    def cutting_high(mod, ii):
        return signal[ii] - mod.power[ii] <= maxpower

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
    model.powerinner = pe.Var(model.ind)


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
        return mod.powerinner[ii] == efficiency_losses + discharge_losses

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
        return mod.energy[ii] == lastenergy + mod.powerinner[ii] * dtimes[ii]

    model.con_integrate = pe.Constraint(model.ind, rule=integrate)


def _cyclic_energy_constraint(model):
    """Ensures that energy content at beginning of interval equals energy
    content at end of interval"""

    model.con_cyclic = pe.Constraint(expr=(model.energyinit ==
                                           model.energy[model.ind.last()]))


def _energy_lower_max_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""

    def energy_lower_max(mod, ii):
        return mod.baseenergy[ii] <= mod.basenergycapacity

    model.con_energylowermax = pe.Constraint(model.ind,
                                             rule=energy_lower_max)

    model.con_energyinitlowermax = \
        pe.Constraint(expr=(model.energyinit <= model.baseenergycapacity))
