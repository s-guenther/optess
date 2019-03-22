#!/usr/bin/env python3

import operator
from overload import overload
import numpy as np
from matplotlib import pyplot as plt
import pickle
from random import randint

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

    def concat(self, other, plausibility_check=True):
        if plausibility_check:
            # TODO implement as warnings
            dt1 = np.mean(self.dtimes)
            dt2 = np.mean(other.dtimes)
            if dt1/dt2 > 10 or dt2/dt1 > 10:
                print('Warning: Sampling Rates of signals differ by a factor '
                      'of 10 or more.')
            v1 = np.mean(self.vals)
            v2 = np.mean(other.vals)
            if v1/v2 > 10 or v2/v1 > 10:
                print('Warning: Magnitudes of signals differ by a factor of '
                      '10 or more.')
        t1 = self.times
        t2 = other.times + t1[-1]
        tt = np.concatenate([t1, t2])
        vv = np.concatenate([self.vals, other.vals])
        return Signal(tt, vv)

    def append(self, other, plausibility_check=True):
        """alias for concat"""
        return self.concat(other, plausibility_check)

    @classmethod
    def multi_concat(cls, signals, plausibility_check=True):
        if plausibility_check:
            # TODO implement as warnings
            dts = [np.mean(signal.dtimes) for signal in signals]
            if max(dts)/min(dts) > 10:
                print('Warning: Sampling Rates of signals differ by a factor '
                      'of 10 or more')
            vs = [np.mean(signal.vals) for signal in signals]
            if max(vs)/min(vs) > 10:
                print('Warning: Magnitudes of signals differ by a factor of '
                      '10 or more.')
        offsets = np.cumsum([signal.times[-1] for signal in signals])
        offsets = np.insert(offsets[:-1], 0, 0)
        tt = np.concatenate([signal.times + offset
                             for signal, offset in zip(signals, offsets)])
        vv = np.concatenate([signal.vals for signal in signals])
        return Signal(tt, vv)

    def burst_split(self, n):
        """Evenly splits signal into n new signals of equal length (within
        integer rounding precision)."""
        npoints = len(self)
        cuts = np.linspace(0, npoints, n+1)
        cuts[-1] += 1
        cuts = np.array(np.round(cuts), dtype=int)
        starts = cuts[:-1]
        ends = cuts[1:]
        offinds = (starts - 1)[1:]
        offsets = np.concatenate([[0], [self.times[ind] for ind in offinds]])
        signals = list()
        for start, end, offset in zip(starts, ends, offsets):
            times, vals = self[start:end]
            signals.append(Signal(times-offset, vals))
        return signals

    def equals(self, other):
        """Returns True if two signals are equal, and False if not.
        Does not return an element-wise comparision of two signals as __eq__
        does."""
        try:
            other = Signal(other)
        except AttributeError:
            return False
        try:
            res = self == other
        except (TimeValueVectorsNotEqualLengthError,
                TimeVectorsNotEqualError):
            return False
        return all(res.vals)

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
        times, vals = self[:]
        times = np.insert(times, 0, 0)
        vals = np.insert(vals, 0, vals[0])
        pltfcn(times, vals, **kwargs)
        plt.draw()

    def pprint(self):
        # TODO implement
        pass

    def save(self, filename=None):
        if filename is None:
            tag = '{:04x}'.format(randint(0, 16**4-1))
            template = 'signal_{}_n_{}_t_{}'
            fields = (tag, len(self), self.times[-1])
            filename = template.format(*fields)
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'sig'

        with open(sep.join([filename, fileend]), 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'sig'

        with open(sep.join([filename, fileend]), 'rb') as file:
            sig = pickle.load(file)
        return sig

    def _validate(self):
        is_strictly_monotoneous = all(val > 0 for val in self.dtimes)
        if not is_strictly_monotoneous:
            raise TimeVectorNotStrictlyMonotoneousError

        is_same_length = len(self.times) == len(self.vals)
        if not is_same_length:
            raise TimeValueVectorsNotEqualLengthError

    def _is_same_time(self, other):
        if len(self) == len(other):
            return all([a == b for a, b in zip(self.times, other.times)])
        else:
            return False

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
