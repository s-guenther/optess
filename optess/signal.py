#!/usr/bin/env python3

import operator
from overload import overload
import numpy as np
from matplotlib import pyplot as plt

from .utility import make_empty_axes


class TimeValueVectorsNotEqualLengthError(ValueError):
    pass


class TimeVectorNotStrictlyMonotoneousError(ValueError):
    pass


class TimeVectorsNotEqualError(ValueError):
    pass


class Signal:
    @overload
    def __init__(self, times, vals):
        self._times = times if type(times) is np.ndarray else np.array(times)
        self._vals = vals if type(vals) is np.ndarray else np.array(vals)

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
        prependzero = np.concatenate(([0], self.times))
        return prependzero[1:] - prependzero[:-1]

    @property
    def amv(self):
        """Average mean value"""
        return 1/self.times[-1]*sum(self.vals*self.dtimes)

    @property
    def arv(self):
        """Average rectified value"""
        return 1/self.times[-1]*sum(abs(self.vals)*self.dtimes)

    @property
    def rms(self):
        """Root Mean Square"""
        return np.sqrt(1/self.times[-1]*sum(self.vals**2*self.dtimes))

    @property
    def form(self):
        """Form Factor"""
        return self.rms/self.arv

    @property
    def crest(self):
        """Crest Factor"""
        return max(self.vals)/self.rms

    def integrate(self, int_constant=0):
        """Integrates the signal and returns the integral as a new signal."""
        energies = np.cumsum(self.vals*self.dtimes) + int_constant
        return Signal(self.times, energies)

    def cycles(self, capacity=None):
        """Calculates the equivalent cycle load (discharged energy divided
        by capacity. If capacity is not provided, it is estimated from
        signal itself. Note that this is a approximation and note that the
        cycle parameter is only useful for a certain class of signals."""
        def discharged(signal):
            neg_parts = signal*(signal <= 0)
            return min(neg_parts.integrate().vals)

        def capacity_from_signal(signal):
            integral = signal.integrate()
            return max(integral.vals) - min(integral.vals)

        dis = -discharged(self)
        cap = capacity if capacity else capacity_from_signal(self)

        return dis/cap

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
            valvec = list(fcn(val, other) for val in self.vals)
        else:
            # Signal function is applied
            if not self._is_same_time(other):
                raise TimeVectorsNotEqualError
            valvec = list(fcn(a, b) for a, b in zip(self.vals, other.vals))
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

    def __abs__(self):
        return Signal(self.times, abs(self.vals))

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
