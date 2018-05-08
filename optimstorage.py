#!/usr/bin/env/python3
"""Defines data structure 'Signal'"""

from collections import namedtuple
from overload import overload
from enum import Enum


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
        self._discharge = None
        self.power = power
        self._efficiency = efficiency
        self._discharge = selfdischarge

    @__init__.add
    def __init__(self, storage):
        self.__init__(storage.power, storage.efficiency, storage.discharge)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        try:
            self._power = _Power(float(value), float(value))
        except TypeError:
            self._power = _Power(float(value[0]), float(value[1]))

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        try:
            self._efficiency = _Efficiency(float(value), float(value))
        except TypeError:
            self._efficiency = _Efficiency(float(value[0]), float(value[1]))

    @property
    def selfdischarge(self):
        return self._discharge

    @selfdischarge.setter
    def selfdischarge(self, value):
        self._discharge = float(value)

    def pprint(self):
        # TODO implement
        pass

    def pplot(self):
        # TODO implement
        pass

    def __str__(self):
        # Todo implement
        pass


class Signal:
    @overload
    def __init__(self, time, val):
        self._time = tuple(float(stime) for stime in time)
        self._val = tuple(float(sval) for sval in val)
        self._dtime = [second - first for first, second
                       in zip((0,) + self.time[:-1], self.time)]

        self._validate()

    @__init__.add
    def __init__(self, signal):
        self.__init__(signal.time, signal.val)

    @property
    def time(self):
        return self._time

    @property
    def val(self):
        return self._val

    @property
    def dtime(self):
        return self._dtime

    def pplot(self):
        # TODO implement
        pass

    def pprint(self):
        # TODO implement
        pass

    def _validate(self):
        is_strictly_monotoneous = all(val > 0 for val in self.dtime)
        if not is_strictly_monotoneous:
            raise TimeVectorNotStrictlyMonotoneousError

        is_same_length = len(self.time) == len(self.val)
        if not is_same_length:
            raise TimeValueVectorsNotEqualLengthError

    def __getitem__(self, item):
        return self.time[item], self.val[item]

    def __add__(self, other):
        """Adds values of two signals if time base is equivalent"""
        is_same_time = all([a == b for a, b in zip(self.time, other.time)])
        if not is_same_time:
            raise TimeVectorsNotEqualError

        vals = (a + b for a, b in zip(self.val, other.val))

        return Signal(self.time, vals)

    def __neg__(self):
        return Signal(self.time, [-x for x in self.val])

    def __len__(self):
        return len(self.time)

    def __str__(self):
        # TODO implement
        pass


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
        self.type = _ObjectiveType(value)

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
    gurobi = 1
    glpk = 2


class Strategy(Enum):
    """Gathers allowed strategies for storage operation"""
    inter = 1
    nointer = 2


class Results:
    """Represents results of optimization in an easily accassible way"""
    def __init__(self, model, signal):
        """Writes results in pyomo model to optimstorage classes."""
        # Direct variables stored in model
        signalvarnames = ['base', 'peak',
                          'baseplus', 'baseminus', 'peakplus', 'peakminus',
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
                    Signal(signal.time,
                           getattr(model, varname).get_values().values()))

        for varname in floatvarnames:
            setattr(self,
                    varname,
                    getattr(model, varname).get_values().values())

        # Derived variables
        self.both = self.base + self.peak
        self.bothinner = self.baseinner + self.peakinner

        self.baseinter = self.inter
        self.peakinter = -self.inter

        self.baseinterplus = Signal(signal.time,
                                    (x if x > 0 else 0
                                     for x in self.baseinter.val))
        self.baseinterminus = Signal(signal.time,
                                     (x if x <= 0 else 0
                                      for x in self.baseinter.val))
        self.peakinterplus = Signal(signal.time,
                                    (x if x > 0 else 0
                                     for x in self.peakinter.val))
        self.peakinterminus = Signal(signal.time,
                                     (x if x <= 0 else 0
                                      for x in self.peakinter.val))

        self.baseinnerplus = Signal(signal.time,
                                    (x if x <= 0 else 0
                                     for x in self.baseinner.val))
        self.baseinnerminus = Signal(signal.time,
                                     (x if x <= 0 else 0
                                      for x in self.baseinner.val))

        self.peakinnerplus = Signal(signal.time,
                                    (x if x <= 0 else 0
                                     for x in self.peakinner.val))
        self.peakinnerminus = Signal(signal.time,
                                     (x if x <= 0 else 0
                                      for x in self.peakinner.val))

    def pprint(self, fig=100):
        # TODO implement
        pass

    def pplot(self):
        # TODO implement
        pass

    def __str__(self):
        # TODO implement
        pass
