#!/usr/bin/env python3
"""This module defines an IdealHybridOptBuilder class, which builds the
optimization model of a hybrid storage without losses in various flavours.
It is used by the hybrid storage optimization class through composition."""

import pyomo.environ as pe
from typing import Tuple, Optional, Union
from datetime import datetime
from warnings import warn
from numbers import Real

from .signal import Signal
from .target import Target, TargetError
from .target import TEXACT, TPOWER, TAPPROX, TENERGY
from .storage import Storage, MAX_TAU
from .singleoptbuilder import MAX_RAMP, MAX_CYCLES


_NOT_ALL_DEF_ERR_MSG = \
    'Inputs storage and target must be completely defined - ' \
    'storage.energy, storage.power, target.val must not be ' \
    'None (except for target.type is \'exact\').'


class InconsistentInputError(ValueError):
    pass


# Class defines program logic, functions provide interface to pyomo
class HybridOptBuilder:
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
            fmt = 'IdealHybridOptBuilder-%y%m%d-%H%M%S'
            self.name = datetime.now().strftime(fmt)

        self._base = None
        self._peak = None
        self._target = None
        self._inter = None
        self._model = None

    def min_energy(self, base: Storage, peak: Storage, target: Target,
                   inter: bool = True, interideal: bool = False,
                   max_cycles: Optional[Real] = None,
                   max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                   nametag: str = 'Minimize-Energy'):
        # Check input
        [is_e, is_p, is_t] = self._get_defined_state(base, peak, target)
        if not (is_p and is_t):
            msg = 'You have to define base.power, peak.power and target.val ' \
                  'for Objective \'min_energy\'.'
            raise InconsistentInputError(msg)
        if is_e:
            warn('Storage object is overdefined. '
                 'Ignoring storage.energy parameter.')

        self._reset_model(base, peak, target, inter, interideal, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_power(self._model)
        _fix_target(self._model)
        _min_energy_obj(self._model)

        return self._model

    def min_power(self, base: Storage, peak: Storage, target: Target,
                  power_disch_ch_factor: Union[Real, Tuple[Real, Real]] = 1,
                  inter: bool = True, interideal: bool = False,
                  max_cycles: Optional[Real] = None,
                  max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                  nametag: str = 'Minimize-Power'):
        """power_disch_ch_factor can be a scalar number, than, both base and
        peak storage have the same discharge/charge ratio. If a tuple is
        passed, it can be set individually instead."""
        # Check input
        [is_e, is_p, is_t] = self._get_defined_state(base, peak, target)
        if not (is_e and is_t):
            msg = 'You have to define base.energy, peak.energy and ' \
                  'target.val for Objective \'min_power\'.'
            raise InconsistentInputError(msg)
        if is_p:
            warn('Storage object is overdefined. '
                 'Ignoring storage.power parameter.')

        self._reset_model(base, peak, target, inter, interideal, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_energy(self._model)
        _fix_target(self._model)
        # TODO ensure that always a tuple is passed to _min_power_obj
        if isinstance(power_disch_ch_factor, Real):
            power_disch_ch_factor = [power_disch_ch_factor]*2
        _min_power_obj(self._model, power_disch_ch_factor)

    def min_target(self, base: Storage, peak: Storage, target: Target,
                   inter: bool = True, interideal: bool = False,
                   max_cycles: Optional[Real] = None,
                   max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                   nametag='Minimize-Target'):
        # Check input
        [is_e, is_p, is_t] = self._get_defined_state(base, peak, target)
        if not (is_e and is_p):
            msg = 'You have to define storage.energy and storage.power for ' \
                  'Objective \'min_target\'.'
            raise InconsistentInputError(msg)
        if is_t and target.type is not TEXACT:
            warn('Target object is overdefined. '
                 'Ignoring target.val parameter.')

        self._reset_model(base, peak, target, inter, interideal, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_energy(self._model)
        _fix_power(self._model)
        _min_target_obj(self._model)

    def min_cycles(self, base: Storage, peak: Storage, target: Target,
                   inter: bool = True, interideal: bool = False,
                   max_cycles: Optional[Real] = None,
                   max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                   nametag: str = 'Minimize-Cycles'):
        self._verify_complete_input(base, peak, target)

        self._reset_model(base, peak, target, inter, interideal, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_cycles_obj(self._model)

    def min_ramp(self, base: Storage, peak: Storage, target: Target,
                 inter: bool = True, interideal: bool = False,
                 ramp_disch_ch_factor: Real = 1,
                 max_cycles: Optional[Real] = None,
                 max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                 nametag: str = 'Minimize-Ramp'):
        self._verify_complete_input(base, peak, target)

        self._reset_model(base, peak, target, inter, interideal, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_ramp_obj(self._model, ramp_disch_ch_factor)

    def min_avg_dynamics(self, base: Storage, peak: Storage, target: Target,
                         inter: bool = True, interideal: bool = False,
                         max_cycles: Optional[Real] = None,
                         max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                         nametag: str = 'Minimize-Average-Dynamics'):
        self._verify_complete_input(base, peak, target)

        self._reset_model(base, peak, target, inter, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_avg_dyn_obj(self._model)

    def min_dynamics(self, base: Storage, peak: Storage, target: Target,
                     inter: bool = True, interideal: bool = False,
                     max_cycles: Optional[Real] = None,
                     max_ramp: Optional[Real, Tuple[Real, Real]] = None,
                     ramp_disch_ch_factor: Real = 1,
                     nametag: str = 'Minimize-Dynamics'):
        self._verify_complete_input(base, peak, target)

        self._reset_model(base, peak, target, inter, interideal, nametag)
        self._build_common_model(max_cycles, max_ramp)

        _fix_all(self._model)
        _min_dynamics_obj(self._model, ramp_disch_ch_factor)

        # Build model

    def _reset_model(self, base, peak, target, inter, inter_ideal, name=''):
        self._base = base
        self._peak = peak
        self._target = target
        self._inter = inter
        self._interideal = inter_ideal
        self._model = pe.ConcreteModel(name=name + '-' + self.name)

    def _build_common_model(self, max_cycles=None, max_ramp=None):
        self._initialize_model()
        self._add_common_vars()
        self._add_common_constraints()
        self._define_inter_storage_power_flow()
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
    def _get_defined_state(base, peak, target):
        """Check if Energy, Power, TargetVal is existent and return boolean
        array [is_e, is_p, is_t]"""
        is_e = bool(base.energy) and bool(peak.energy)
        is_p = bool(base.power) and bool(peak.power)
        is_t = bool(target.val) or target.type is TEXACT
        return is_e, is_p, is_t

    def _verify_complete_input(self, base, peak, target):
        """Raises an error InconsistentInputError if input is incomplete."""
        is_all_def = all(self._get_defined_state(base, peak, target))
        if not is_all_def:
            raise InconsistentInputError(_NOT_ALL_DEF_ERR_MSG)

    def _initialize_model(self):
        model = self._model
        model._signal = self.signal
        model._base = self._base
        model._peak = self._peak
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

    def _define_inter_storage_power_flow(self):
        is_storageideal = self._base.is_ideal() and self._peak.is_ideal()
        is_inter = self._inter
        is_interideal = self._interideal
        if not is_inter:
            self._prohibit_power_flow()
        elif not is_storageideal and is_inter and is_interideal:
            self._allow_power_flow_without_losses()
        else:
            self._allow_power_flow_with_losses()

    def _prohibit_power_flow(self):
        _prohibit_inter_power_flow(self._model)

    def _allow_power_flow_without_losses(self):
        _prohibit_inter_power_flow(self._model)
        _allow_inter_power_flow_without_losses(self._model)

    @staticmethod
    def _allow_power_flow_with_losses():
        """No extra constraints needed, so this function is empty"""
        pass

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


def __rule_split_basepower(mod, ii):
    return mod.basepowerplus[ii] + mod.basepowerminus[ii] == mod.basepower[ii]


def __rule_split_peakpower(mod, ii):
    return mod.peakpowerplus[ii] + mod.peakpowerminus[ii] == mod.peakpower[ii]


# ### main
def _power_vars(model, penaltyfactor=1e-3):
    # power over time, inner equals the integrated part where losses have
    # been excluded
    model.power = pe.Var(model.ind)
    model.base = pe.Var(model.ind)
    model.peak = pe.Var(model.ind)
    model.inner = pe.Var(model.ind)
    model.baseinner = pe.Var(model.ind)
    model.peakinner = pe.Var(model.ind)
    # power capacity positive and negative, define as variable as it is not
    # known for now if it is a constraint/bound or an objective
    model.powercapplus = pe.Var(bounds=(0, None))
    model.basecapplus = pe.Var(bounds=(0, None))
    model.peakcapplus = pe.Var(bounds=(0, None))
    model.powercapminus = pe.Var(bounds=(None, 0))
    model.basecapminus = pe.Var(bounds=(None, 0))
    model.peakcapminus = pe.Var(bounds=(None, 0))
    # split power, neccessary for efficiency + and -
    model.powerplus = pe.Var(model.ind, bounds=(0, None))
    model.baseplus = pe.Var(model.ind, bounds=(0, None))
    model.peakplus = pe.Var(model.ind, bounds=(0, None))
    model.powerminus = pe.Var(model.ind, bounds=(None, 0))
    model.baseminus = pe.Var(model.ind, bounds=(None, 0))
    model.peakminus = pe.Var(model.ind, bounds=(None, 0))
    model.con_split_power = pe.Constraint(model.ind, rule=__rule_split_power)
    model.con_split_basepower = \
        pe.Constraint(model.ind, rule=__rule_split_power)
    model.con_split_peakpower = \
        pe.Constraint(model.ind, rule=__rule_split_power)
    # penalty term in objective to ensure that one of powerplus and
    # powerminus is always zero:
    model.objexpr += sum(model.powerplus[ii] - model.powerminus[ii]
                         for ii in model.ind)*penaltyfactor
    model.objexpr += sum(model.basepowerplus[ii] - model.basepowerminus[ii]
                         for ii in model.ind)*penaltyfactor
    model.objexpr += sum(model.peakpowerplus[ii] - model.peakpowerminus[ii]
                         for ii in model.ind)*penaltyfactor


# ### main
def _energy_vars(model):
    # Energy content as function of time
    model.baseenergy = pe.Var(model.ind, bounds=(0, None))
    model.peakenergy = pe.Var(model.ind, bounds=(0, None))
    # Energy capacity (maximum storable energy), as variable until it is
    # known if it is a bound by target or an objective
    model.baseenergycapacity = pe.Var(bounds=(0, None))
    model.peakenergycapacity = pe.Var(bounds=(0, None))
    # Initial condition of energy content
    model.baseenergyinit = pe.Var(bounds=(0, None))
    model.peakenergyinit = pe.Var(bounds=(0, None))


# ### main
def _target_vars(model):
    model.targetvallow = pe.Var()
    model.targetval = pe.Var()


# _add_common_constraints() ###################################################
# noinspection PyProtectedMember
def __baselosses(mod, ii):
    eta_ch = mod._base.efficiency.charge
    eta_dis = mod._base.efficiency.discharge
    tau = mod._base.selfdischarge
    selfdis = 1/tau if tau <= MAX_TAU else 0

    eff_power = (mod.baseplus[ii]*eta_ch + mod.baseminus[ii]/eta_dis)
    lastenergy = mod.baseenergy[ii-1] if ii is not 0 else mod.baseenergyinit
    dis_power = -lastenergy*selfdis

    return mod.baseinner[ii] == eff_power + dis_power


# noinspection PyProtectedMember
def __peaklosses(mod, ii):
    eta_ch = mod._peak.efficiency.charge
    eta_dis = mod._peak.efficiency.discharge
    tau = mod._peak.selfdischarge
    selfdis = 1/tau if tau <= MAX_TAU else 0

    eff_power = (mod.peakpowerplus[ii]*eta_ch + mod.peakpowerminus[ii]/eta_dis)
    lastenergy = mod.peakenergy[ii-1] if ii is not 0 else mod.peakenergyinit
    dis_power = -lastenergy*selfdis

    return mod.peakinner[ii] == eff_power + dis_power


# ### main
def _loss_model(model):
    model.con_baselosses = pe.Constraint(model.ind, rule=__baselosses)
    model.con_peaklosses = pe.Constraint(model.ind, rule=__peaklosses)


def __rule_baseintegrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    lastenergy = mod.baseenergy[ii-1] if ii is not 0 else mod.baseenergyinit
    return mod.baseenergy[ii] == lastenergy + mod.baseinner[ii]*dtimes[ii]


def __rule_peakintegrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    lastenergy = mod.peakenergy[ii-1] if ii is not 0 else mod.peakenergyinit
    return mod.peakenergy[ii] == lastenergy + mod.peakinner[ii]*dtimes[ii]


# ### main
def _integrate_power_constraint(model):
    """Euler Integration per timestep of power to gain energy"""
    model.con_baseintegrate = \
        pe.Constraint(model.ind, rule=__rule_baseintegrate)
    model.con_peakintegrate = \
        pe.Constraint(model.ind, rule=__rule_peakintegrate)


def __rule_base_lower_max(mod, ii):
    return mod.base[ii] <= mod.basecapplus


def __rule_peak_lower_max(mod, ii):
    return mod.peak[ii] <= mod.peakcapplus


def __rule_base_higher_min(mod, ii):
    return mod.basecapminus <= mod.base[ii]


def __rule_peak_higher_min(mod, ii):
    return mod.peakcapminus <= mod.peak[ii]


# ### main
def _power_bound_constraint(model):
    model.con_baselowermax = \
        pe.Constraint(model.ind, rule=__rule_base_lower_max)
    model.con_basehighermin = \
        pe.Constraint(model.ind, rule=__rule_base_higher_min)
    model.con_peaklowermax = \
        pe.Constraint(model.ind, rule=__rule_peak_lower_max)
    model.con_peakhighermin = \
        pe.Constraint(model.ind, rule=__rule_peak_higher_min)


def __rule_baseenergy_lower_max(mod, ii):
    return mod.baseenergy[ii] <= mod.baseenergycapacity


def __rule_peakenergy_lower_max(mod, ii):
    return mod.peakenergy[ii] <= mod.peakenergycapacity


# ### main
def _energy_bound_constraint(model):
    """Ensures that energy capacity of storages is not exceeded at all times"""
    model.con_baseenergylowermax = \
        pe.Constraint(model.ind, rule=__rule_baseenergy_lower_max)
    model.con_baseenergyinitlowermax = \
        pe.Constraint(expr=(model.baseenergyinit <= model.baseenergycapacity))
    model.con_peakenergylowermax = \
        pe.Constraint(model.ind, rule=__rule_baseenergy_lower_max)
    model.con_peakenergyinitlowermax = \
        pe.Constraint(expr=(model.peakenergyinit <= model.peakenergycapacity))


# _add_cycle_constraint() and _add_ramp_constraint() ##########################

# ### main
# noinspection PyProtectedMember
def _max_cycle_constraint(model, max_cycles_tuple):
    dtimes = model._signal.dtimes
    if max_cycles_tuple[0] < MAX_CYCLES:
        basecycle_expr = (sum(-model.baseminus[ii]*dtimes[ii]
                          for ii in model.ind) <=
                          model.baseenergycapacity*max_cycles_tuple[0])
        model.con_max_basecycle = pe.Constraint(expr=basecycle_expr)
    if max_cycles_tuple[1] > -MAX_CYCLES:
        peakcycle_expr = (sum(-model.peakminus[ii]*dtimes[ii]
                          for ii in model.ind) <=
                          model.energycapacity*max_cycles_tuple[0])
        model.con_max_peakcycle = pe.Constraint(expr=peakcycle_expr)


# noinspection PyProtectedMember
def __rule_fixed_baseramp_minus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod._max_ramp_tuple[0] <= mod.base[0] - mod.base[ii]
        else:
            return pe.Constraint.Skip
    return mod._max_ramp_tuple[0] <= mod.base[ii+1] - mod.base[ii]


# noinspection PyProtectedMember
def __rule_fixed_baseramp_plus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod.base[0] - mod.base[ii] <= mod._max_ramp_tuple[1]
        else:
            return pe.Constraint.Skip
    return mod.base[ii+1] - mod.base[ii] <= mod._max_ramp_tuple[1]


# noinspection PyProtectedMember
def __rule_fixed_peakramp_minus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod._max_ramp_tuple[2] <= mod.peak[0] - mod.peak[ii]
        else:
            return pe.Constraint.Skip
    return mod._max_ramp_tuple[2] <= mod.peak[ii+1] - mod.peak[ii]


# noinspection PyProtectedMember
def __rule_fixed_peakramp_plus(mod, ii):
    if ii == mod.ind.last():
        if hasattr(mod, 'con_cyclic'):
            return mod.peak[0] - mod.peak[ii] <= mod._max_ramp_tuple[3]
        else:
            return pe.Constraint.Skip
    return mod.peak[ii+1] - mod.peak[ii] <= mod._max_ramp_tuple[3]


# ### main
def _max_ramp_constraint(model, max_ramp_4_tuple):
    """4-tuple --> [-baseramp, +baseramp, -peakramp, +peakramp]"""
    model._max_ramp_tuple = max_ramp_4_tuple
    if max_ramp_4_tuple[0] > -MAX_RAMP:
        model.con_fixed_baseramp_minus = \
            pe.Constraint(model.ind, rule=__rule_fixed_baseramp_minus)
    if max_ramp_4_tuple[1] > MAX_RAMP:
        model.con_fixed_baseramp_plus = \
            pe.Constraint(model.ind, rule=__rule_fixed_baseramp_plus)
    if max_ramp_4_tuple[2] > -MAX_RAMP:
        model.con_fixed_peakramp_minus = \
            pe.Constraint(model.ind, rule=__rule_fixed_peakramp_minus)
    if max_ramp_4_tuple[3] > MAX_RAMP:
        model.con_fixed_peakramp_plus = \
            pe.Constraint(model.ind, rule=__rule_fixed_peakramp_plus)


# _define_inter_storage_power_flow ############################################

def __rule_prohibit_inter_plus(mod, ii):
    return mod.powerplus[ii] == mod.baseplus[ii] + mod.peakplus[ii]


def __rule_prohibit_inter_minus(mod, ii):
    return mod.powerminus[ii] == mod.baseminus[ii] + mod.peakminus[ii]


# ### main
def _prohibit_inter_power_flow(model):
    model.con_prohibit_inter_plus = \
        pe.Constraint(model.ind, rule=__rule_prohibit_inter_plus)
    model.con_prohibit_inter_minus = \
        pe.Constraint(model.ind, rule=__rule_prohibit_inter_minus)


# ### main
def _allow_inter_power_flow_with_losses(model):
    # Nothing to do, this is covered automatically
    pass


def __rule_baseinterintegrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    lastenergy = mod.baseenergy[ii-1] if ii is not 0 else mod.baseenergyinit
    return (mod.baseenergy[ii] == lastenergy +
                                  (mod.baseinner[ii] + mod.inter)*dtimes[ii])


def __rule_peakinterintegrate(mod, ii):
    # noinspection PyProtectedMember
    dtimes = mod._signal.dtimes
    lastenergy = mod.peakenergy[ii-1] if ii is not 0 else mod.peakenergyinit
    return (mod.peakenergy[ii] == lastenergy +
                                  (mod.peakinner[ii] - mod.inter)*dtimes[ii])


def __rule_baseinter_lower_max(mod, ii):
    return mod.base[ii] + mod.inter[ii] <= mod.basecapplus


def __rule_peakinter_lower_max(mod, ii):
    return mod.peak[ii] - mod.inter[ii] <= mod.peakcapplus


def __rule_baseinter_higher_min(mod, ii):
    return mod.basecapminus <= mod.base[ii] + mod.inter[ii]


def __rule_peakinter_higher_min(mod, ii):
    return mod.peakcapminus <= mod.peak[ii] - mod.inter[ii]


# ### main
def _allow_inter_power_flow_without_losses(model):
    # Prohibit inter power flow with losses
    _prohibit_inter_power_flow(model)
    # Allow it through an additional variable
    model.inter = pe.Var(model.ind)
    # Remove old integration constraints
    model.del_component(model.con_baseintegrate)
    model.del_component(model.con_peakintegrate)
    # Integrate
    model.con_baseintegrate = \
        pe.Constraint(model.ind, rule=__rule_baseinterintegrate)
    model.con_peakintegrate = \
        pe.Constraint(model.ind, rule=__rule_peakinterintegrate)
    # Bounds with inter
    model.con_baselowermax = \
        pe.Constraint(model.ind, rule=__rule_baseinter_lower_max)
    model.con_basehighermin = \
        pe.Constraint(model.ind, rule=__rule_baseinter_higher_min)
    model.con_peaklowermax = \
        pe.Constraint(model.ind, rule=__rule_peakinter_lower_max)
    model.con_peakhighermin = \
        pe.Constraint(model.ind, rule=__rule_peakinter_higher_min)


# _add_target() ###############################################################
# Only a selection following functions are applied to the model.

# ### main
def _cyclic_constraint(model):
    model.con_cyclicbase = \
        pe.Constraint(expr=(model.baseenergyinit ==
                            model.baseenergy[model.ind.last()]))
    model.con_cyclicpeak = \
        pe.Constraint(expr=(model.peakenergyinit ==
                            model.peakenergy[model.ind.last()]))


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
def _min_power_obj(model, disch_ch_factor: Real = 1):
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
