#!/usr/bin/env python3

from collections import namedtuple
from enum import Enum
from overload import overload


class Solver(Enum):
    """Gathers allowed solvers for simulation"""
    # TODO add more solvers
    gurobi = 'gurobi'
    glpk = 'glpk'


class Strategy(Enum):
    """Gathers allowed strategies for storage operation"""
    inter = 'inter'
    nointer = 'nointer'


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
