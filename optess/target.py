#!/usr/bin/env python3

from collections import namedtuple
from enum import Enum
import numbers


class Solver(Enum):
    """Gathers allowed solvers for simulation"""
    # TODO add more solvers
    gurobi = 'gurobi'
    glpk = 'glpk'


class ObjectiveError(ValueError):
    pass


class Objective(Enum):
    """Determine the objective which shall be minimized. """
    min_e = 'min_e'
    min_p = 'min_p'
    min_t = 'min_t'


MIN_E = Objective.min_e
MIN_P = Objective.min_p
MIN_T = Objective.min_t


class TargetType(Enum):
    """power - peak power shaving; energy - limit energy taken from grid;
    exact - follow, approx - follow approximately"""
    power = 'power'
    energy = 'energy'
    exact = 'exact'
    approx = 'approx'


TEXACT = TargetType.exact
TENERGY = TargetType.energy
TPOWER = TargetType.power
TAPPROX = TargetType.approx


class TargetError(Exception):
    pass


class Target:
    def __init__(self, ttype: TargetType = TEXACT, val=None):
        """Depending on Target Type, the val argument must be None,
        a number, or an iterable of two numbers. Internally, this is always
        converted to two numbers in self.vals. Property self.val always
        returns self.vals[1]."""
        self._type = ttype
        self._vals = self._check_val_input(val)

    @property
    def type(self):
        return self._type

    @property
    def val(self):
        """Returns the second, upper value of the vals 2-tuple. In case of
        type==exact, this has no meaning, in case of type==energy,
        it returns the max energy, in case of type==power or type==approx,
        this returns the upper bound of the target (which is likely to be more
        relevant as the lower one is not set or the same.)"""
        return self._vals[1]

    @property
    def vals(self):
        """Always returns a 2-tuple. Elements might be without meaning,
        depending on target type."""
        return self._vals

    def _check_val_input(self, value):
        vals = [0, 0]
        if self.type is TEXACT and value is not None:
            raise TargetError('TargetType.exact takes no value argument.')
        elif self.type is TENERGY and value is not None:
            if isinstance(value, numbers.Number):
                vals = [value, value]
            else:
                msg = 'TargetType.energy takes exactly one number as value ' \
                      'argument (if a variable is passed).'
                raise TargetError(msg)
        elif ((self.type is TPOWER or self.type is TAPPROX)
              and value is not None):
            if isinstance(value, numbers.Number):
                vals = ([-1e99, value] if self.type is TPOWER else
                        [-value, value])
            else:
                tstring = ('TargetType.power' if self.type is TPOWER else
                           'Targettype.approx')
                msg = '{} takes one number as value ' \
                      'argument or an iterable of two numbers (if a ' \
                      'variable is passed) where v1 <= v2'.format(tstring)
                try:
                    is_two_numbers = len(value) == 2 and \
                                     isinstance(value[0], numbers.Number) and \
                                     isinstance(value[1], numbers.Number)

                    if is_two_numbers and value[0] <= value[1]:
                        self._val = value[0], value[1]
                    else:
                        raise TargetError(msg)
                except Exception:
                    raise TargetError(msg)
        else:
            raise TargetError('Unknown TargetType')
        return vals

    # TODO add class functions _generate_from_rel_target(signal)
