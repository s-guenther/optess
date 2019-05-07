#!/usr/bin/env python3
"""This module defines an IdealSingleOptBuilder class, which builds the
optimization model of a single storage without losses in various flavours.
It is used by the single storage optimization class through composition."""

import pyomo.environ as pe
from typing import AnyStr
from datetime import datetime
from warnings import warn

from .signal import Signal
from .target import Target, Objective, ObjectiveError, TargetError
from .target import TEXACT, TPOWER, TAPPROX, TENERGY
from .target import MIN_E, MIN_P, MIN_T
from .storage import IdealStorage


class InconsistentInputError(ValueError):
    pass


class IdealSingleOptBuilder:
    """Builder class which returns an object able to generate various
    flavours of a model of a single storage optimization with an ideal
    storage. This is a special case of SingleOptBuilder with better
    performance for ideal storages."""
    def __init__(self, signal: Signal, name: AnyStr = None):
        """"""
        self.signal = signal

        if name:
            self.name = name
        else:
            fmt = 'IdealSingleOptBuilder-%y%m%d-%H%M%S'
            self.name = datetime.now().strftime(fmt)

        self._storage = None
        self._target = None
        self._objective = None
        self._model = None

    def _clear_protected(self):
        self._storage = None
        self._target = None
        self._objective = None
        self._model = None

    def build_stage_1(self, storage: IdealStorage, target: Target,
                      objective: Objective):
        """First stage minimizes the objective, neglecting cycles and
        dynamic stress. Make sure the definition of storage and target is
        consistent and only two of 3 parameters (storage.power,
        storage.energy, target.val) are defined (leave one empty with
        'None')."""
        self._check_2_of_3_defined(storage, target, objective)
        self._clear_protected()

        self._storage = storage
        self._target = target
        self._objective = objective
        self._model = pe.ConcreteModel(name='Model-1st-stage-' + self.name)

        self._initialize_model()
        self._add_common_vars()
        self._add_common_constraints()
        self._add_target()
        self._add_objective()

        return self._model

    def build_stage_2(self, storage: IdealStorage, target: Target):
        """Second stage minimizes cycles, neglecting dynamic stress. It
        expects that the model was solved before with stage 1 and the
        minimized objective is known and passed (But you can omit the
        first stage if you already know the minimized objective from other
        considerations or you are not interested in minimizing it before).
        Make sure that storage and target are fully defined."""
        self._check_all_defined(storage, target)
        self._clear_protected()

    def build_stage_3(self, storage: IdealStorage, target: Target,
                      cycles=None):
        """Third stage minimizes the dynamic stress. It expects that the
        model was solved before with stage 1 and the minimized objective is
        known and passed and with stage two and the minimized cycles are
        passed. But you can omit the first stage if you are not interested
        in minimizing the objective before or already know it from other
        calculations and you can also omit the second stage if you are not
        interested in minimizing the cycles before."""
        self._check_all_defined(storage, target)
        self._clear_protected()

    def build(self, *args, **kwargs):
        """Alias for build_stage_1"""
        return self.build_stage_1(*args, **kwargs)

    def build_stage_minimize_cycles(self, *args, **kwargs):
        """Alias for build_stage_2"""
        return self.build_stage_2(*args, **kwargs)

    def build_stage_minimize_dynamics(self, *args, **kwargs):
        """Alias for build_stage_3"""
        return self.build_stage_3(*args, **kwargs)

    @staticmethod
    def _get_defined_state(storage, target):
        is_e = bool(storage.energy)
        is_p = bool(storage.power)
        is_t = bool(target.val) or target.type is TEXACT
        is_all = all([is_e, is_p, is_t])
        return is_e, is_p, is_t, is_all

    @classmethod
    def _check_all_defined(cls, storage, target):
        all_defined = cls._get_defined_state(storage, target)[3]
        if not all_defined:
            msg = 'Inputs storage and target must be completely defined - ' \
                  'storage.energy, storage.power, target.val must not be ' \
                  'None (except for target.type is \'exact\').'
            raise InconsistentInputError(msg)
        return None

    @classmethod
    def _check_2_of_3_defined(cls, storage, target, objective):
        """Checks the input definitions of storage, target and objective and
        throw warnings if overdefined and errors if ill
        defined/underdefined."""
        is_e, is_p, is_t, is_all = cls._get_defined_state(storage, target)

        if objective is Objective.min_e:
            if is_all:
                warn('Model inputs are overdefined but consistent. Ignoring '
                     'storage.energy parameter.')
            elif is_t and is_p:
                pass
            else:
                msg = 'You have to define storage.power and target.val for ' \
                      'Objective \'min_e\'.'
                raise InconsistentInputError(msg)
        elif objective is Objective.min_p:
            if is_all:
                warn('Model inputs are overdefined but consistent. Ignoring '
                     'storage.power parameter.')
            elif is_t and is_e:
                pass
            else:
                msg = 'You have to define storage.energy and target.val for ' \
                      'Objective \'min_p\'.'
                raise InconsistentInputError(msg)
        elif objective is Objective.min_z:
            if target.type is not TEXACT:
                if is_all:
                    warn('Model inputs are overdefined but consistent. '
                         'Ignoring target.val parameter.')
                elif is_p and is_e:
                    pass
                else:
                    msg = 'You have to define storage.energy and ' \
                          'storage.power for Objective \'min_z\'.'
                    raise InconsistentInputError(msg)
            else:
                if is_p and is_e:
                    warn('Model inputs are overdefined but consistent. For '
                         'Target type \'exact\' you only have to define one '
                         'storage parameter. Ignoring storage.energy '
                         'parameter.')
                elif is_p or is_e:
                    pass
                else:
                    msg = 'You have to define storage.energy or ' \
                          'storage.power for Objective \'min_z\' and ' \
                          'Target \'exact\'.'
                    raise InconsistentInputError(msg)
        else:
            raise ObjectiveError('Unknown Objective {}'.format(objective))
        return None

    def _initialize_model(self):
        model = self._model
        model._signal = self.signal
        model._storage = self._storage
        model._target = self._target
        model._objective = self._objective

        _initialize_model(self._model)

    def _add_common_vars(self):
        """Initialize all (common) variables, do set bounds if known a
        priori, do not set bounds if they must be specified by target and/or
        objective."""
        _power_vars(self._model)
        _energy_vars(self._model)
        _target_vars(self._model)

    def _add_common_constraints(self):
        """Constraints that are shard by all objectives and targets."""
        _integrate_power_constraint(self._model)
        _power_bound_constraint(self._model)
        _energy_bound_constraint(self._model)

    def _add_target(self):
        """This adds target specific constraints and eventually new auxilliary
        variables to define targets or penalty terms to objective."""
        if self._target.type is TPOWER:
            _peak_cutting_target(self._model)
            _cyclic_constraint(self._model)
        elif self._target.type is TENERGY:
            _energy_consumption_target(self._model)
            _cyclic_constraint(self._model)
        elif self._target.type is TEXACT:
            _exact_target(self._model)
        elif self._target.type is TAPPROX:
            _approximate_target(self._model)
        else:
            raise TargetError('Unknown Target {}'.format(self._target))

    def _add_objective(self):
        """This may also add some bounds or constraints to variables."""
        objective = self._objective
        if objective is MIN_E:
            _min_energy_obj(self._model)
        elif objective is MIN_P:
            _min_power_obj(self._model)
        elif objective is MIN_T:
            _min_target_obj(self._model)
        else:
            raise ObjectiveError('Unknown Objective {}'.format(objective))


# ###
# ### Pyomo model building functions
# ###

# _initialize_model() #########################################################

def _initialize_model(model):
    # Set for range of variables
    # noinspection PyProtectedMember
    model.ind = pe.Set(initialize=range(len(model._signal)), ordered=True)
    # Create empty objective expression, which is filled in subsequent steps
    model.objexpr = 0


# _add_vars() #################################################################

# ### main
def _power_vars(model):
    # power over time
    model.power = pe.Var(model.ind)
    # power capacity positive and negative
    model.powercapplus = pe.Var(bounds=(0, None))
    model.powercapminus = pe.Var(bounds=(None, 0))


# ### main
def _energy_vars(model):
    # Energy content as function of time
    model.energy = pe.Var(model.ind, bounds=(0, None))
    # Energy capacity (maximum storable energy)
    model.energycapacity = pe.Var(bounds=(0, None))
    # Initial condition of base and peak energy content
    model.energyinit = pe.Var(bounds=(0, None))


# ### main
def _target_vars(model):
    model.targetvallow = pe.Var()
    model.targetval = pe.Var()


# _add_common_constraints() ###################################################

def __integrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    if ii is 0:
        # Exception handling for first step, initial condition
        lastenergy = mod.energyinit
    else:
        lastenergy = mod.energy[ii - 1]
    return mod.energy[ii] == lastenergy + mod.power[ii] * dtimes[ii]


# ### main
def _integrate_power_constraint(model):
    """Euler Integration per timestep of power to gain energy"""
    model.con_integrate = pe.Constraint(model.ind, rule=__integrate)


def __power_lower_max(mod, ii):
    return mod.power[ii] <= mod.powercapplus


def __power_higher_min(mod, ii):
    return mod.powercapminus <= mod.power[ii]


# ### main
def _power_bound_constraint(model):
    model.con_powerlowermax = \
        pe.Constraint(model.ind, rule=__power_lower_max)
    model.con_powerhighermin = \
        pe.Constraint(model.ind, rule=__power_higher_min)


def __energy_lower_max(mod, ii):
    return mod.energy[ii] <= mod.energycapacity


# ### main
def _energy_bound_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""
    model.con_energylowermax = \
        pe.Constraint(model.ind, rule=__energy_lower_max)
    model.con_energyinitlowermax = \
        pe.Constraint(expr=(model.energyinit <= model.energycapacity))


# _add_target() ###############################################################
# Only a selection following functions are applied to the model.

# ### main
def _cyclic_constraint(model):
    model.con_cyclic = pe.Constraint(expr=(model.energyinit ==
                                           model.energy[model.ind.last()]))


# ### Target Peak Cutting #####################################

# noinspection PyProtectedMember
def __cutting_low(mod, ii):
    signal = mod._signal.vals
    minpower = mod._target.val[0]
    return signal[ii] + mod.power[ii] >= mod.targetvallow


# noinspection PyProtectedMember
def __cutting_high(mod, ii):
    signal = mod._signal.vals
    maxpower = mod._target.val[1]
    return signal[ii] + mod.power[ii] <= mod.targetval


# ### main
def _peak_cutting_target(model):
    """This target will limit the maximum power taken from grid to a certain
    value. Can also limit the maximum power fed into grid or the minimum power
    taken from grid."""
    model.con_cutting_low = pe.Constraint(model.ind, rule=__cutting_low)
    model.con_cutting_high = pe.Constraint(model.ind, rule=__cutting_high)


# ### Target Energy Reduction/Consumption #####################

def __split_delta(mod, ii):
    # noinspection PyProtectedMember
    signal = mod._signal.vals
    return (signal[ii] + mod.power[ii] ==
            mod.deltaplus[ii] + mod.deltaminus[ii])


# ### main
def _energy_consumption_target(model):
    """This target will decrease the amount of energy taken from grid/supply
    etc. It will also add constraints to reach objective. This objective only
    makes sense if the power from grid changes sign, i.e. if power is fed into
    grid"""
    dtimes = model.signal.dtimes
    maxenergy = model.target.val

    model.deltaplus = pe.Var(model.ind, bounds=(0, None))
    model.deltaminus = pe.Var(model.ind, bounds=(None, 0))

    # TODO assume that this is not neccessary --> prove it
    # Ensures that one of deltaplus/deltaminus is zero
    # model.objexpr += sum(model.deltaplus[ii] - model.deltaminus[ii]
    #                      for ii in model.ind)*multiplier

    model.con_split_delta = \
        pe.Constraint(model.ind, rule=__split_delta)

    model.con_limit_energy = \
        pe.Constraint(expr=sum(model.deltaplus[ii]*dtimes[ii]
                               for ii in model.ind) <= model.targetval)


# ### Target Follow Exact #####################################

def __follow_exact(mod, ii):
    # noinspection PyProtectedMember
    signal = mod._signal.vals
    return signal[ii] + mod.power[ii] == 0


# ### main
def _exact_target(model):
    model.con_exact = pe.Constraint(model.ind, rule=__follow_exact)


# ### Target Follow Semiexact/Approximately ##################

# ### main
def _approximate_target(model):
    """This is equal to _exact_target() in the formulation, but not in the
    intention. Also for target approximately, no cyclic constraint is
    applied."""
    model.con_approx_low = pe.Constraint(model.ind, rule=__cutting_low)
    model.con_approx_high = pe.Constraint(model.ind, rule=__cutting_high)


# _add_objective() ############################################################
# Only a selection following functions are applied to the model.

# noinspection PyProtectedMember
def _fix_target(model):
    model.targetvallow.fix(model._target.vals[0])
    model.targetval.fix(model._target.vals[1])


# noinspection PyProtectedMember
def _fix_power(model):
    model.power.setlb(model._storage.power.min)
    model.power.setub(model._storage.power.max)
    model.con_powerlowermax.deactivate()
    model.con_powerhighermin.deactivate()
    model.powercapminus.fix(model._storage.power.min)
    model.powercapplus.fix(model._storage.power.max)


# noinspection PyProtectedMember
def _fix_energy(model):
    model.energy.setub(model._storage.energy)
    model.con_energylowermax.deactivate()
    model.energycapacity.fix(model._storage.energy)


# ### Objective Minimize Energy ##############################

# ### main
# noinspection PyProtectedMember
def _min_energy_obj(model):
    _fix_power(model)
    _fix_target(model)
    model.objexpr += model.energycapacity
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Power ###############################

# ### main
def _min_power_obj(model):
    _fix_energy(model)
    _fix_target(model)
    model.objexpr += model.powercapplus - model.powercapminus
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Target ##############################

# ### main
# noinspection PyProtectedMember
def _min_target_obj(model):
    _fix_energy(model)
    _fix_power(model)
    model.objexpr += model.targetval
    if model._target.type is TEXACT:
        _fix_target(model)
    elif model._target.type is TENERGY:
        model.targetvallow.fix(0)
    elif model._target.type is TPOWER:
        model.targetvallow.fix(-1e99)
    elif model._target.type is TAPPROX:
        model.con_approx_equality = \
            pe.Constraint(expr=(model.targetval == model.targetvallow))
    else:
        raise TargetError('Unknown Target {}'.format(model._target.type))
    model.objexpr += model.targetval
    model.obj = pe.Objective(expr=model.objexpr)

