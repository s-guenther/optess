#!/usr/bin/env python3
"""Various factories to define data and optimization setups"""

import numpy as np
import scipy.interpolate as interp
from collections import namedtuple

from .signal import Signal
from .storage import Storage
from .objective import Objective


class DataFactory:
    @staticmethod
    def std(npoints=110):
        """Alternative Testcase providing simple load profile, best choose
        a whole multiple of 22"""
        x = range(23)
        y = [3, 5, 2, 5, 3, 5, 2, 5, 3, 5, 2, 5,
             3, 1, 2, 3, 2, 1, 2, 2, 2, 2, 2]
        spl = interp.interp1d(x, y, 'linear')
        xx = np.linspace((len(x)-1)/npoints, len(x)-1, npoints)
        yy = spl(xx)
        return Signal(xx, yy)

    @staticmethod
    def alt(npoints=64):
        """Standard Testcase providing simple load profile"""
        x = range(0, 17)
        y = [3, 5, 3, 2, 3, 4, 3, 0, -1, 0, 3, 5, 3, -2, 3, 2, 2]
        # noinspection PyTypeChecker
        spl = interp.PchipInterpolator(x, y)
        xx = np.linspace((len(x)-1)/npoints, len(x)-1, npoints)
        yy = spl(xx)
        return Signal(xx, yy)

    @staticmethod
    def rand(npoints=256, mu=10, freq=(3, 8, 10, 50), amp=(1, 1.5, 2, 2),
             time=None, seed=None, interpolate='linear'):
        if time is None:
            time = npoints
        if seed is None:
            seed = np.random.randint(0, int(1e6))
            print('Randomly chosen seed is {}.'.format(seed))
        vals = mu*np.ones(npoints)
        np.random.seed(seed)
        for f, a in zip(freq, amp):
            y = np.random.randn(2*f*time+2)*a
            x = np.linspace(0, npoints-1, len(y))
            inter = interp.interp1d(x, y, interpolate)
            valsinter = inter(range(npoints))
            vals += valsinter
        return Signal(np.linspace(time/npoints, time, npoints), vals)

    @staticmethod
    def distorted_sin(npoints=256, mu=10, freq=(3, 8, 10, 50),
                      amp=(1, 1.5, 2, 2), ampvariance=0.3, jittervariance=0.3,
                      time=1, seed=None, interpolate='linear'):
        """Superposes multiple sin waves, where the phase is chosen randomly
        and amplitude and frequency show small deviations, defined by sigma1
        and sigma2"""
        if seed is None:
            seed = np.random.randint(0, int(1e6))
            print('Randomly chosen seed is {}.'.format(seed))
        try:
            # noinspection PyTypeChecker
            iter(ampvariance)
        except TypeError:
            ampvariance = (ampvariance,) * len(amp)
        try:
            # noinspection PyTypeChecker
            iter(jittervariance)
        except TypeError:
            jittervariance = (jittervariance,) * len(freq)
        np.random.seed(seed)
        vals = mu*np.ones(npoints)
        times = np.linspace(time/npoints, time, npoints)
        for f, a0, s, j in zip(freq, amp, ampvariance, jittervariance):
            support = int(np.ceil(4*f*time) + 1)
            phase = np.random.rand()*2*np.pi
            aa = a0 + s*a0*np.random.randn(support)
            tt = np.linspace(0, time, support)
            ainter = interp.interp1d(tt, aa, interpolate)
            ampvariation = ainter(times)*np.sin(2*np.pi*f*times + phase)

            supportf = int(np.ceil(f*time/4 + 1))
            ttf = np.linspace(0, time, supportf)
            randf = np.random.randn(supportf)*j + f
            ff = abs(randf) + 1e-6
            dttinter = interp.interp1d(ttf, 1/ff, interpolate)
            timesvaried = np.cumsum(dttinter(times))
            timesvaried = timesvaried*time/timesvaried[-1]

            allinter = interp.interp1d(timesvaried, ampvariation, interpolate,
                                       fill_value='extrapolate')
            vals += allinter(times)

        return Signal(times, vals)

    @classmethod
    def freq(cls, npoints=1000, mu=10, nfreqs=32,
             freqsupport=(0.05, 1.5, 1.6, 3, 3.1, 4.5),
             ampsupport=(0.9, 1, 0.2, 0.2, 1, 0.9),
             ampvar=0.0, ampjitter=0.3, freqjitter=0.05,
             time=100, seed=None, interpolate='linear', spacing=np.linspace):
        """Similar to distorted sin, but takes a frequency response (as
        freq-amp pairs as input and builds randomized signal which roughly
        matches this frequency response"""
        if seed is None:
            seed = np.random.randint(0, int(1e6))
            print('Randomly chosen seed is {}.'.format(seed))
        try:
            # noinspection PyTypeChecker
            iter(ampvar)
        except TypeError:
            ampvar = (ampvar,) * len(ampsupport)
        np.random.seed(seed)

        ampinterp = interp.interp1d(freqsupport, ampsupport,
                                    kind=interpolate, fill_value='extrapolate')
        ampvarinterp = interp.interp1d(freqsupport, ampvar, interpolate,
                                       fill_value='extrapolate')

        freqs = spacing(freqsupport[0], freqsupport[-1], num=nfreqs)
        amps = (ampinterp(freqs) +
                ampvarinterp(freqs)*np.random.rand(*freqs.shape))

        return cls.distorted_sin(npoints, mu, freqs, amps, ampjitter,
                                 freqjitter, time, seed, interpolate)


class StorageFactory:
    """Builds a storage with predefined loss models. Currently implemented
    functions: ideal, low, medium, high. All functions take power as input
    argument (default = 1)"""
    @staticmethod
    def ideal(power=1.0):
        losses = [1, np.inf]
        return Storage(power, *losses)

    @staticmethod
    def low(power=1.0):
        losses = [0.95, 1/0.001]
        return Storage(power, *losses)

    @staticmethod
    def medium(power=1.0):
        losses = [0.9, 1/0.002]
        return Storage(power, *losses)

    @staticmethod
    def high(power=1.0):
        losses = [0.8, 1/0.005]
        return Storage(power, *losses)


class ObjectiveFactory:
    """Builds a predefined objective as input data for the optimisation.
    Currently implemented: std03
    """
    @staticmethod
    def std03():
        return Objective('power', 3)


class HybridSetupFactory:
    """Defines a complete setup for a hybrid storage optimisation"""
    @staticmethod
    def std(cut=0.5, losses=StorageFactory.low, npoints=110, strategy='inter'):
        # This tuple can be unpacked directly into OptimModel
        HybridSetup = namedtuple('HybridSetup', 'signal base peak objective '
                                                'strategy solver name')
        signal = DataFactory.std(npoints)
        singlepower = 2
        base = losses(cut*singlepower)
        peak = losses((1-cut)*singlepower)
        objective = ObjectiveFactory.std03()
        solver = 'gurobi'
        name = 'Hybrid Storage Optimization ' \
               '{}.{}.{}.{}.{}'.format('std', base.power, peak.power,
                                       objective.type, strategy)
        return HybridSetup(signal, base, peak,
                           objective, strategy, solver, name)


class SingleSetupFactory:
    @staticmethod
    def std(losses=StorageFactory.low, npoints=110):
        # This tuple can be unpacked directly into OptimModel
        SingleSetup = namedtuple('SingleSetup', 'signal storage objective '
                                                'solver name')
        signal = DataFactory.std(npoints)
        storage = losses(2)
        objective = ObjectiveFactory.std03()
        solver = 'gurobi'
        name = 'Single Storage Optimization ' \
               '{}.{}.{}'.format('std', storage.power, objective.type)
        return SingleSetup(signal, storage, objective, solver, name)


def randspace(start, stop, npoints, jitter=0.5):
    if jitter >= 1 or jitter < 0:
        raise ValueError('Deviation must be in [0, 1)')
    linpoints = np.linspace(start, stop, npoints)
    dt = linpoints[1] - linpoints[0]
    dev = (np.random.rand(npoints)*dt-dt/2) * jitter
    randpoints = linpoints + dev
    randpoints[0] = start
    randpoints[-1] = stop
    return randpoints

