#!/usr/bin/env python3

import operator
from overload import overload
from matplotlib import pyplot as plt

from utility import make_empty_axes


class TimeValueVectorsNotEqualLengthError(ValueError):
    pass


class TimeVectorNotStrictlyMonotoneousError(ValueError):
    pass


class TimeVectorsNotEqualError(ValueError):
    pass


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
        ax = ax if ax else make_empty_axes()
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
