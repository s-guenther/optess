#!/usr/bin/env python3
"""This module defines an IdealSingleOptBuilder class, which builds the
optimization model of a single storage without losses in various flavours.
It is used by the single storage optimization class through composition."""

import pyomo.environ as pe
from typing import Tuple, Optional
from datetime import datetime
from warnings import warn
from numbers import Real

from .signal import Signal
from .target import Target, TargetError
from .target import TEXACT, TPOWER, TAPPROX, TENERGY
from .storage import Storage


# Defines the maximum time constant for self discharge, higher values are
# treated as infinity, i.e. ideal storage without self discharge
# Built in for numerical stability and better conditioning of matrices
MAX_TAU = 1e10

_NOT_ALL_DEF_ERR_MSG = \
    'Inputs storage and target must be completely defined - ' \
    'storage.energy, storage.power, target.val must not be ' \
    'None (except for target.type is \'exact\').'


class InconsistentInputError(ValueError):
    pass


# Class defines program logic, functions provide interface to pyomo
class SingleOptBuilder:
    """Builder class which returns an object able to generate various
    flavours of a model of a single storage optimization with an ideal
    storage. This is a special case of SingleOptBuilder with better
    performance for ideal storages."""
    def __init__(self, signal: Signal, name: Optional[str] = None):
        """"""
        self.signal = signal

        if name:
            self.name = name
        else:
            fmt = 'IdealSingleOptBuilder-%y%m%d-%H%M%S'
            self.name = datetime.now().strftime(fmt)

        self._storage = None
        self._target = None
        self._model = None

    def min_energy(self, storage: Storage, target: Target,
                   max_cycles: Optional[Real] = None,
                   max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                   nametag: str = 'Minimize-Energy'):
        # Check input
        [is_e, is_p, is_t] = self._get_defined_state(storage, target)
        if not (is_p and is_t):
            msg = 'You have to define storage.power and target.val for ' \
                  'Objective \'min_energy\'.'
            raise InconsistentInputError(msg)
        if is_e:
            warn('Storage object is overdefined. '
                 'Ignoring storage.energy parameter.')

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_power(self._model)
        _fix_target(self._model)
        _min_energy_obj(self._model)

        return self._model

    def min_power(self, storage: Storage, target: Target,
                  power_disch_ch_factor: Real = 1,
                  max_cycles: Optional[Real] = None,
                  max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                  nametag: str = 'Minimize-Power'):
        # Check input
        [is_e, is_p, is_t] = self._get_defined_state(storage, target)
        if not (is_e and is_t):
            msg = 'You have to define storage.energy and target.val for ' \
                  'Objective \'min_power\'.'
            raise InconsistentInputError(msg)
        if is_p:
            warn('Storage object is overdefined. '
                 'Ignoring storage.power parameter.')

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_energy(self._model)
        _fix_target(self._model)
        _min_power_obj(self._model, power_disch_ch_factor)

    def min_target(self, storage: Storage, target: Target,
                   max_cycles: Optional[Real] = None,
                   max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                   nametag='Minimize-Target'):
        # Check input
        [is_e, is_p, is_t] = self._get_defined_state(storage, target)
        if not (is_e and is_p):
            msg = 'You have to define storage.energy and storage.power for ' \
                  'Objective \'min_target\'.'
            raise InconsistentInputError(msg)
        if is_t and target.type is not TEXACT:
            warn('Target object is overdefined. '
                 'Ignoring target.val parameter.')

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_energy(self._model)
        _fix_power(self._model)
        _min_target_obj(self._model)

    def min_cycles(self, storage: Storage, target: Target,
                   max_cycles: Optional[Real] = None,
                   max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                   nametag: str = 'Minimize-Cycles'):
        self._verify_complete_input(storage, target)

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_cycles_obj(self._model)

    def min_ramp(self, storage: Storage, target: Target,
                 ramp_disch_ch_factor: Real = 1,
                 max_cycles: Optional[Real] = None,
                 max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                 nametag: str = 'Minimize-Ramp'):
        self._verify_complete_input(storage, target)

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_ramp_obj(self._model, ramp_disch_ch_factor)

    def min_avg_dynamics(self, storage: Storage, target: Target,
                         max_cycles: Optional[Real] = None,
                         max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                         nametag: str = 'Minimize-Average-Dynamics'):
        self._verify_complete_input(storage, target)

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_avg_dyn_obj(self._model)

    def min_dynamics(self, storage: Storage, target: Target,
                     ramp_disch_ch_factor: Real = 1,
                     max_cycles: Optional[Real] = None,
                     max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                     nametag: str = 'Minimize-Dynamics'):
        self._verify_complete_input(storage, target)

        self._reset_model(storage, target, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_dynamics_obj(self._model, ramp_disch_ch_factor)

        # Build model

    def _reset_model(self, storage, target, name=''):
        self._storage = storage
        self._target = target
        self._model = pe.ConcreteModel(name=name + '-' + self.name)

    def _build_common_model(self, max_cycles=None, max_ramp=None):
        self._initialize_model()
        self._add_common_vars()
        self._add_common_constraints()
        # Add target performs different operations depending on target type
        self._add_target()
        # max_cycles and max_ramp constraint is not reasonable for all
        # objectives, therefore, each objective must choose if it wants to
        # set these constraints by passing the constraint values or leaving
        # it as None
        if max_cycles:
            self._add_cycle_constraint(max_cycles)
        if max_ramp:
            self._add_ramp_constraint(max_ramp)

    @staticmethod
    def _get_defined_state(storage, target):
        """Check if Energy, Power, TargetVal is existent and return boolean
        array [is_e, is_p, is_t]"""
        is_e = bool(storage.energy)
        is_p = bool(storage.power)
        is_t = bool(target.val) or target.type is TEXACT
        return is_e, is_p, is_t

    def _verify_complete_input(self, storage, target):
        """Raises an error InconsistentInputError if input is incomplete."""
        is_all_def = all(self._get_defined_state(storage, target))
        if not is_all_def:
            raise InconsistentInputError(_NOT_ALL_DEF_ERR_MSG)

    def _initialize_model(self):
        model = self._model
        model._signal = self.signal
        model._storage = self._storage
        model._target = self._target

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
        elif self._target.type is TENERGY:
            _energy_consumption_target(self._model)
        elif self._target.type is TEXACT:
            _exact_target(self._model)
        elif self._target.type is TAPPROX:
            _approximate_target(self._model)
        else:
            raise TargetError('Unknown Target {}'.format(self._target))

        if self._target.is_cyclic:
            _cyclic_constraint(self._model)

    def _add_cycle_constraint(self, max_cycles):
        _max_cycle_constraint(self._model, max_cycles)

    def _add_ramp_constraint(self, max_ramp):
        if not isinstance(max_ramp, tuple):
            max_ramp = [-max_ramp, max_ramp]
        if max_ramp[0] >= 0:
            raise ValueError('First max_ramp parameter must be negative.')
        _max_ramp_constraint(self._model, max_ramp)


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


# _add_common_vars() ##########################################################

def __rule_split_power(mod, ii):
    return mod.powerplus[ii] + mod.powerminus[ii] == mod.power[ii]


# ### main
def _power_vars(model, penaltyfactor=1e-3):
    # power over time, inner equals the integrated part where losses have
    # been excluded
    model.power = pe.Var(model.ind)
    model.inner = pe.Var(model.ind)
    # power capacity positive and negative, define as variable as it is not
    # known for now if it is a constraint/bound or an objective
    model.powercapplus = pe.Var(bounds=(0, None))
    model.powercapminus = pe.Var(bounds=(None, 0))
    # split power, neccessary for efficiency + and -
    model.powerplus = pe.Var(model.ind, bounds=(0, None))
    model.powerminus = pe.Var(model.ind, bounds=(None, 0))
    model.con_split_power = pe.Constraint(model.ind, rule=__rule_split_power)
    # penalty term in objective to ensure that one of powerplus and
    # powerminus is always zero:
    model.objexpr += sum(model.powerplus[ii] - model.powerminus[ii]
                         for ii in model.ind)*penaltyfactor


# ### main
def _energy_vars(model):
    # Energy content as function of time
    model.energy = pe.Var(model.ind, bounds=(0, None))
    # Energy capacity (maximum storable energy), as variable until it is
    # known if it is a bound by target or an objective
    model.energycapacity = pe.Var(bounds=(0, None))
    # Initial condition of energy content
    model.energyinit = pe.Var(bounds=(0, None))


# ### main
def _target_vars(model):
    model.targetvallow = pe.Var()
    model.targetval = pe.Var()


# _add_common_constraints() ###################################################
# noinspection PyProtectedMember
def __losses(mod, ii):
    eta_ch = mod._storage.efficiency.charge
    eta_dis = mod._storage.efficiency.discharge
    tau = mod._storage.selfdischarge
    selfdis = 1/tau if tau <= MAX_TAU else 0

    eff_power = (mod.powerplus[ii]*eta_ch + mod.powerminus[ii]/eta_dis)
    lastenergy = mod.energy[ii-1] if ii is not 0 else mod.energyinit
    dis_power = -lastenergy*selfdis

    return mod.inner[ii] == eff_power + dis_power


def __rule_integrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    if ii is 0:
        # Exception handling for first step, initial condition
        lastenergy = mod.energyinit
    else:
        lastenergy = mod.energy[ii - 1]
    return mod.energy[ii] == lastenergy + mod.inner[ii] * dtimes[ii]


# ### main
def _integrate_power_constraint(model):
    """Euler Integration per timestep of power to gain energy"""
    model.con_integrate = pe.Constraint(model.ind, rule=__rule_integrate)


def __rule_power_lower_max(mod, ii):
    return mod.power[ii] <= mod.powercapplus


def __rule_power_higher_min(mod, ii):
    return mod.powercapminus <= mod.power[ii]


# ### main
def _power_bound_constraint(model):
    model.con_powerlowermax = \
        pe.Constraint(model.ind, rule=__rule_power_lower_max)
    model.con_powerhighermin = \
        pe.Constraint(model.ind, rule=__rule_power_higher_min)


def __rule_energy_lower_max(mod, ii):
    return mod.energy[ii] <= mod.energycapacity


# ### main
def _energy_bound_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""
    model.con_energylowermax = \
        pe.Constraint(model.ind, rule=__rule_energy_lower_max)
    model.con_energyinitlowermax = \
        pe.Constraint(expr=(model.energyinit <= model.energycapacity))


# _add_cycle_constraint() and _add_ramp_constraint() ##########################

# ### main
def _max_cycle_constraint(model, max_cycles):
    cycle_expr = (sum(-model.powerminus[ii] for ii in model.ind) <=
                  model.energycapacity*max_cycles)
    model.con_max_cycle = pe.Constraint(expr=cycle_expr)


# noinspection PyProtectedMember
def __rule_fixed_ramp_plus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod.power[0] - mod.power[ii] <= mod._max_ramp_tuple[1]
        else:
            return pe.Constraint.Skip
    return mod.power[ii+1] - mod.power[ii] <= mod._max_ramp_tuple[0]


# noinspection PyProtectedMember
def __rule_fixed_ramp_minus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod._max_ramp_tuple[0] <= mod.power[0] - mod.power[ii]
        else:
            return pe.Constraint.Skip
    return mod._max_ramp_tuple[0] <= mod.power[ii+1] - mod.power[ii]


# ### main
def _max_ramp_constraint(model, max_ramp_tuple):
    model._max_ramp_tuple = max_ramp_tuple
    model.con_fixed_ramp_plus = \
        pe.Constraint(model.ind, rule=__rule_fixed_ramp_plus)
    model.con_fixed_ramp_minus = \
        pe.Constraint(model.ind, rule=__rule_fixed_ramp_minus)


# _add_target() ###############################################################
# Only a selection following functions are applied to the model.

# ### main
def _cyclic_constraint(model):
    model.con_cyclic = pe.Constraint(expr=(model.energyinit ==
                                           model.energy[model.ind.last()]))


# ### Target Peak Cutting #####################################

# noinspection PyProtectedMember
def __rule_cutting_low(mod, ii):
    signal = mod._signal.vals
    return signal[ii] + mod.power[ii] >= mod.targetvallow


# noinspection PyProtectedMember
def __rule_cutting_high(mod, ii):
    signal = mod._signal.vals
    return signal[ii] + mod.power[ii] <= mod.targetval


# ### main
def _peak_cutting_target(model):
    """This target will limit the maximum power taken from grid to a certain
    value. Can also limit the maximum power fed into grid or the minimum power
    taken from grid."""
    model.con_cutting_low = pe.Constraint(model.ind, rule=__rule_cutting_low)
    model.con_cutting_high = pe.Constraint(model.ind, rule=__rule_cutting_high)


# ### Target Energy Reduction/Consumption #####################

def __rule_split_delta(mod, ii):
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
        pe.Constraint(model.ind, rule=__rule_split_delta)

    model.con_limit_energy = \
        pe.Constraint(expr=sum(model.deltaplus[ii]*dtimes[ii]
                               for ii in model.ind) <= maxenergy)


# ### Target Follow Exact #####################################

def __rule_follow_exact(mod, ii):
    # noinspection PyProtectedMember
    signal = mod._signal.vals
    return signal[ii] + mod.power[ii] == 0


# ### main
def _exact_target(model):
    model.con_exact = pe.Constraint(model.ind, rule=__rule_follow_exact)


# ### Target Follow Semiexact/Approximately ##################

# ### main
def _approximate_target(model):
    """This is equal to _exact_target() in the formulation, but not in the
    intention. Also for target approximately, no cyclic constraint is
    applied."""
    model.con_approx_low = pe.Constraint(model.ind, rule=__rule_cutting_low)
    model.con_approx_high = pe.Constraint(model.ind, rule=__rule_cutting_high)


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


def _fix_all(model):
    _fix_target(model)
    _fix_power(model)
    _fix_energy(model)


# ### Objective Minimize Energy ##############################

# ### main
# noinspection PyProtectedMember
def _min_energy_obj(model):
    model.objexpr += model.energycapacity
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Power ###############################

# ### main
def _min_power_obj(model, disch_ch_factor=1):
    ff = disch_ch_factor
    model.con_lock_power_cap = \
        pe.Constraint(expr=(model.powercapplus == model.powercapminus*ff))
    model.objexpr += model.powercapplus - model.powercapminus
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Target ##############################

# ### main
# noinspection PyProtectedMember
def _min_target_obj(model):
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


# ### Objective Minimize Cycles

# ### main
def _min_cycles_obj(model):
    model.objexpr += sum(-model.powerminus[ii] for ii in model.ind)
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Ramp

def __rule_ramp_plus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod.power[0] - mod.power[ii] <= mod.max_ramp
        else:
            return pe.Constraint.Skip
    return mod.power[ii+1] - mod.power[ii] <= mod.max_ramp


# noinspection PyProtectedMember
def __rule_ramp_minus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return -mod.max_ramp*mod._ramp_factor <= mod.power[0] - \
                   mod.power[ii]
        else:
            return pe.Constraint.Skip
    return -mod.max_ramp*mod._ramp_factor <= mod.power[ii+1] - mod.power[ii]


# ### main
def _min_ramp_obj(model, ramp_disch_ch_factor, weight=1):
    model.max_ramp = pe.Var()
    model._ramp_factor = ramp_disch_ch_factor
    model.con_ramp_plus = pe.Expression(model.ind, rule=__rule_ramp_plus)
    model.con_ramp_minus = pe.Expression(model.ind, rule=__rule_ramp_minus)
    # noinspection PyTypeChecker
    model.objexpr += weight*model.max_ramp
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Average Dynamics

def __rule_powerdiff(mod, ii):
    if ii == mod.ind.last():
        return mod.powerdiff[ii] == mod.power[0] - mod.power[ii]
    return mod.powerdiff[ii] == mod.power[ii+1] - mod.power[ii]


def __rule_split_powerdiff(mod, ii):
    return mod.powerdiff[ii] == mod.powerdiffplus[ii] + mod.powerdiffminus[ii]


# ### main
def _min_avg_dyn_obj(model, weight=1):
    model.powerdiff = pe.Var(model.ind)
    model.powerdiffplus = pe.Var(model.ind, bounds=(0, None))
    model.powerdiffminus = pe.Var(model.ind, bounds=(None, 0))
    model.con_powerdiff = \
        pe.Constraint(model.ind, rule=__rule_powerdiff)
    model.con_powerdiffsplit = \
        pe.Constraint(model.ind, rule=__rule_split_powerdiff)
    model.objexpr += weight*sum(model.powerdiffplus[ii] for ii in model.ind)
    model.obj = pe.Objective(expr=model.objexpr)


# ### Objective Minimize Dynamics

# ### main
# noinspection PyProtectedMember
def _min_dynamics_obj(model, ramp_disch_ch_factor, weight=1):
    sig = model._signal - model._signal.amv
    w1 = 1
    w2 = weight/sig.cycles()
    _min_ramp_obj(model, ramp_disch_ch_factor, weight=w1)
    model.del_component(model.obj)
    _min_avg_dyn_obj(model, weight=w2)
