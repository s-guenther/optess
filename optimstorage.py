#!/usr/bin/env/python3
"""Defines data structure 'Signal'"""

import operator
from collections import namedtuple
from enum import Enum
from matplotlib import pyplot as plt
from overload import overload


class TimeValueVectorsNotEqualLengthError(ValueError):
    pass


class TimeVectorNotStrictlyMonotoneousError(ValueError):
    pass


class TimeVectorsNotEqualError(ValueError):
    pass


_Power = namedtuple('Power', 'min max')
_Efficiency = namedtuple('Efficiency', 'charge discharge')


class Storage:
    @overload
    def __init__(self, power, efficiency, selfdischarge):
        self._power = None
        self._efficiency = None
        self._selfdischarge = None
        self.power = power
        self.efficiency = efficiency
        self.selfdischarge = selfdischarge

    @__init__.add
    def __init__(self, storage):
        self.__init__(storage.power, storage.efficiency, storage.selfdischarge)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        try:
            self._power = _Power(-float(value), float(value))
        except TypeError:
            self._power = _Power(float(value[0]), float(value[1]))
        if self._power.min >= self._power.max:
            raise ValueError('Min power lower equal max power')

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        try:
            self._efficiency = _Efficiency(float(value), float(value))
        except TypeError:
            self._efficiency = _Efficiency(float(value[0]), float(value[1]))
        if any(val <= 0 for val in self._efficiency):
            raise ValueError('Efficiency must be greater than zero')

    @property
    def selfdischarge(self):
        return self._selfdischarge

    @selfdischarge.setter
    def selfdischarge(self, value):
        self._selfdischarge = float(value)

    def pprint(self):
        # TODO implement
        pass

    def pplot(self):
        # TODO implement
        pass

    def __repr__(self):
        strfmt = '<{cls}(Power({pwr.min}, {pwr.max}), ' \
                 'Efficiency({eff.charge}, {eff.discharge}), {selfdis})>'
        fields = dict(cls=self.__class__.__name__,
                      pwr=self.power,
                      eff=self.efficiency,
                      selfdis=self.selfdischarge)
        return strfmt.format(**fields)


class Signal:
    @overload
    def __init__(self, times, vals):
        self._times = tuple(float(time) for time in times)
        self._vals = tuple(float(val) for val in vals)
        self._dtimes = [second - first for first, second
                        in zip((0,) + self.times[:-1], self.times)]

        self._validate()

    @__init__.add
    def __init__(self, signal):
        self.__init__(signal.times, signal.vals)

    @property
    def times(self):
        return self._times

    @property
    def vals(self):
        return self._vals

    @property
    def dtimes(self):
        return self._dtimes

    def pplot(self, plotfcn='step', ax=None, **kwargs):
        ax = ax if ax else _make_empty_axes()
        pltfcn = getattr(ax, plotfcn)
        pltfcn(*self[:], **kwargs)
        plt.draw()

    def pprint(self):
        # TODO implement
        pass

    def _validate(self):
        is_strictly_monotoneous = all(val > 0 for val in self.dtimes)
        if not is_strictly_monotoneous:
            raise TimeVectorNotStrictlyMonotoneousError

        is_same_length = len(self.times) == len(self.vals)
        if not is_same_length:
            raise TimeValueVectorsNotEqualLengthError

    def _is_same_time(self, other):
        return all([a == b for a, b in zip(self.times, other.times)])

    def __getitem__(self, item):
        return self.times[item], self.vals[item]

    def _operator_template(self, other, fcn):
        """Defines a template functions for scalar operations like __add__,
        __eq__, __mul__, __lt__, __gt__. The template defines a try except
        else clause which performs control flow depending on scalar or
        signal input. fcn is the function which is performed on the inputs"""
        try:
            # Duck typed test if input is signal
            other = Signal(other)
        except AttributeError:
            # Assume input is scalar, Scalar function is applied
            valvec = (fcn(val, other) for val in self.vals)
        else:
            # Signal function is applied
            if not self._is_same_time(other):
                raise TimeVectorsNotEqualError
            valvec = (fcn(a, b) for a, b in zip(self.vals, other.vals))
        return Signal(self.times, valvec)

    def __add__(self, other):
        """Adds values of two signals if time base is equivalent or add a
        scalar to signal"""
        return self._operator_template(other, fcn=operator.add)

    def __radd__(self, other):
        """Radds values of two signals if time base is equivalent or add a
        signal to a scalar."""
        return self._operator_template(other, fcn=operator.add)

    def __sub__(self, other):
        """Subtracts values of two signals if time base is equivalent or
        substract a scalar from signal"""
        return self._operator_template(other, fcn=operator.sub)

    def __rsub__(self, other):
        """Rsubs values of two signals if time base is equivalent or sub a
        signal from a scalar."""

        return -self._operator_template(other, fcn=operator.add)

    def __eq__(self, other):
        """Elementwise comparision, other can be a signal by itself on the
        same time base, or a scalar"""
        return self._operator_template(other, fcn=operator.eq)

    def __mul__(self, other):
        """Elementwise multiplication in case other is a signal, or scalar
        multiplication in case of other is a scalar"""
        return self._operator_template(other, fcn=operator.mul)

    def __rmul__(self, other):
        """Elementwise multiplication in case other is a signal, or scalar
        multiplication in case of other is a scalar"""
        return self._operator_template(other, fcn=operator.mul)

    def __truediv__(self, other):
        """Devide elementwise or by a scalar"""
        return self._operator_template(other, fcn=operator.truediv)

    def __lt__(self, other):
        """Comparision with other signal of same time base or scalar"""
        return self._operator_template(other, fcn=operator.lt)

    def __gt__(self, other):
        """Comparision with other signal of same time base or scalar"""
        return self._operator_template(other, fcn=operator.gt)

    def __ge__(self, other):
        """Comparision with other signal of same time base or scalar"""
        return self._operator_template(other, fcn=operator.ge)

    def __le__(self, other):
        """Comparision with other signal of same time base or scalar"""
        return self._operator_template(other, fcn=operator.le)

    def __neg__(self):
        return Signal(self.times, [-x for x in self.vals])

    def __len__(self):
        return len(self.times)

    def __repr__(self):
        strfmt = '<<{cls}({times}, {vals})>, length={length}>'
        times = (repr(self._times) if len(self) <= 8
                 else '<times at {}>'.format(hex(id(self._times))))
        vals = (repr(self._vals) if len(self) <= 8
                else '<vals at {}>'.format(hex(id(self.times))))
        fields = dict(cls=self.__class__.__name__,
                      times=times,
                      vals=vals,
                      length=len(self))
        return strfmt.format(**fields)


class _ObjectiveType(Enum):
    """_ObjectiveType is either 'power' or 'energy'. In the first case,
    it cuts power, in the second case, it limits the energy taken from grid"""
    power = 'power'
    energy = 'energy'


_ObjectiveValue = namedtuple('_ObjectiveValue', 'min max')


class Objective:
    """Objective specifies a type and a value"""
    @overload
    def __init__(self, objtype, val):
        self._type = None
        self._val = None
        self.type = objtype
        self.val = val

    @__init__.add
    def __init__(self, objective):
        self.__init__(objective.type, objective.val)

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        try:
            minval, maxval = value
        except TypeError:
            minval, maxval = 0, value
        self._val = _ObjectiveValue(float(minval), float(maxval))

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = _ObjectiveType(value)

    def validate(self, signal):
        """Checks if objective can be even reached theoretically with the
        provided signal (if objective type is power cutting, it cannot be
        cut below the average power"""
        # TODO implement
        pass

    def pprint(self):
        # TODO implement
        pass

    def pplot(self):
        # TODO implement
        pass


class Solver(Enum):
    """Gathers allowed solvers for simulation"""
    # TODO add more solvers
    gurobi = 'gurobi'
    glpk = 'glpk'


class Strategy(Enum):
    """Gathers allowed strategies for storage operation"""
    inter = 'inter'
    nointer = 'nointer'


class HybridResults:
    """Represents results of optimization in an easily accassible way"""
    def __init__(self, model, signal):
        """Writes results in pyomo model to optimstorage classes."""
        # Direct variables stored in model
        signalvarnames = ['base', 'peak',
                          'baseinner', 'peakinner',
                          'inter',
                          'baseenergy', 'peakenergy',
                          'binbaselower', 'binbaseupper',
                          'binpeaklower', 'binpeakupper',
                          'bininterlower', 'bininterupper']
        floatvarnames = ['baseenergycapacity', 'peakenergycapacity',
                         'baseenergyinit', 'peakenergyinit']

        for varname in signalvarnames:
            setattr(self,
                    varname,
                    Signal(signal.times,
                           getattr(model, varname).get_values().values()))

        for varname in floatvarnames:
            setattr(self,
                    varname,
                    list(getattr(model, varname).get_values().values())[0])

        # Derived variables
        self.both = self.base + self.peak
        self.bothinner = self.baseinner + self.peakinner

        self.baseinter = self.inter
        self.peakinter = -self.inter

        self.basesignedlosses = ((self.base - self.baseinner) *
                                 (self.base >= 0) +
                                 (self.baseinner - self.base) *
                                 (self.base < 0))
        self.peaksignedlosses = ((self.peak - self.peakinner) *
                                 (self.peak >= 0) +
                                 (self.peakinner - self.peak) *
                                 (self.peak < 0))

    def pprint(self, fig=100):
        # TODO implement
        print(self.__dict__)

    def pplot(self, ax=None):
        # TODO pass additional arguments to plot functions ?
        ax = ax if ax else _make_empty_axes()

        # Define functions which extract pos/neg vals from a signal and a
        # function to apply these functions to a list of signals
        def get_pos_vals(signal):
            return ((signal >= 0)*signal).vals

        def get_neg_vals(signal):
            return ((signal < 0)*signal).vals

        def apply_fcn_to_signals(inputsignals, fcn):
            return [fcn(signal) for signal in inputsignals]

        # Calculate plotdata with helper functions
        signals = [self.baseinner, self.peakinner,
                   self.basesignedlosses, self.peaksignedlosses,
                   self.baseinter, self.peakinter]
        posvalvecs = apply_fcn_to_signals(signals, get_pos_vals)
        negvalvecs = apply_fcn_to_signals(signals, get_neg_vals)
        timevec = self.both.times

        # Plot positive and negative part of stackplot separately
        plotconfig = dict(step='pre',
                          colors=('#548b54', '#8b1a1a',  # palegreen, firebrick
                                  '#7ccd7c', '#cd2626',  # 4 - 3 - 1
                                  '#9aff9a', '#ff3030'))
        ax.stackplot(timevec, *posvalvecs, **plotconfig)
        ax.stackplot(timevec, *negvalvecs, **plotconfig)

        # add black zero line
        ax.axhline(color='black')
        # add both base/peak added
        self.both.pplot(ax=ax)

    def __repr__(self):
        strfmt = '<<{cls} at {resid}>, base={b}, peak={p}>'
        fields = dict(cls=self.__class__.__name__,
                      resid=hex(id(self)),
                      b=self.baseenergycapacity,
                      p=self.peakenergycapacity)
        return strfmt.format(**fields)


class NoResults:
    """Dummy Class which is returned if the solver failed"""
    pass


def _make_empty_axes():
    ax = plt.figure().add_subplot(1, 1, 1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')
    return ax
