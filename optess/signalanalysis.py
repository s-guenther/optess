#!/usr/bin/env python3
"""Functions and Methods which analyze a signal but are not included in the
signal class itself, as they are a little bit more complex."""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as grdspc
from operator import itemgetter
from scipy.signal import resample, windows
import warnings

from .signal import Signal
from .utility import make_empty_axes


class BinsMustBeIntegerDivisorError(ValueError):
    pass


class PEHMap:
    def __init__(self, power, energy, bins=(20, 20)):
        bins = (bins[1], bins[0])
        x = energy.vals
        y = power.vals
        xyrange = ((0, max(x)), (-max(abs(y)), max(abs(y))))
        weights = power.dtimes
        self.map, self.eedges, self.pedges = \
            np.histogram2d(x, y, bins, xyrange, weights=weights)
        self.map = self.map/sum(self.map.flatten())
        self.eweights = np.sum(self.map, 1)
        self.pweights = np.sum(self.map, 0)

    def pplot(self, ax=None, linecolor='b', mapcolor='Blues'):
        # TODO rework ax logic
        add_subplots_at_end = False
        if not ax:
            fig = plt.figure()
            maingrid = grdspc.GridSpec(1, 1)
            inner = grdspc.GridSpecFromSubplotSpec(2, 2,
                                                   subplot_spec=maingrid[0],
                                                   width_ratios=[3, 1],
                                                   height_ratios=[1, 3])
            ax = [plt.Subplot(fig, ele) for ele in itemgetter(0, 2, 3)(inner)]
            add_subplots_at_end = True
        else:
            fig = plt.gcf()

        ax2, ax1, ax3 = ax
        ax1.set_xlabel('Energy')
        ax1.set_ylabel('Power')

        extent = (min(self.eedges), max(self.eedges),
                  min(self.pedges), max(self.pedges))
        ax1.imshow(self.map.T, aspect='auto', extent=extent, origin='lower',
                   cmap=plt.get_cmap(mapcolor), interpolation='nearest')
        ax2.step(self.eedges, np.append(0, self.eweights), color=linecolor)
        ax2.autoscale(tight=True)
        ax3.step(np.append(self.pweights, 0), self.pedges, color=linecolor)
        ax3.autoscale(tight=True)
        if add_subplots_at_end:
            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            fig.add_subplot(ax3)

    def rebin(self, bins=(5, 5)):
        """TODO rebin only works if the number of new bins are an integer
        TODO divisor of the original number of bins"""
        bins = (bins[1], bins[0])
        binsorig = np.array(self.map.shape)
        if any(bo % b for bo, b in zip(binsorig, bins)):
            raise BinsMustBeIntegerDivisorError

        joinx = int(len(self.eweights) / bins[0])
        joiny = int(len(self.pweights) / bins[1])
        xedges = np.linspace(self.eedges[0], self.eedges[-1], bins[0] + 1)
        yedges = np.linspace(self.pedges[0], self.pedges[-1], bins[1] + 1)
        xweights = np.sum(np.reshape(self.eweights, [-1, joinx]), 1)
        yweights = np.sum(np.reshape(self.pweights, [-1, joiny]), 1)
        hmap = np.reshape(self.map, [binsorig[0], bins[1], -1])
        hmap = np.sum(hmap, 2)
        hmap = np.reshape(hmap.T, [bins[1], bins[0], -1])
        hmap = np.sum(hmap, 2).T

        return self._init_with_data(hmap, xedges, yedges, xweights, yweights)

    @classmethod
    def _init_with_data(cls, hmap, xedges, yedges, xweights, yweights):
        self = cls.__new__(cls)
        self.map, self.eedges, self.pedges = hmap, xedges, yedges
        self.eweights, self.pweights = xweights, yweights
        return self


class FFT:
    def __init__(self, signal, window=windows.hann):
        """Please use scipy.signal.window windows instead of numpy windows.
        Refer to http://www.ni.com/white-paper/4278/en/ to choose a correct
        window"""
        signal = self.equidistant(signal)
        (self.freq_scale, self.ampl_scale, self.amv) = self.fft_info(signal)
        self.fft = self.fft(signal - self.amv, window=window)

    @property
    def amplitude(self):
        return abs(self.fft)*self.ampl_scale

    @property
    def phase(self):
        return np.angle(self.fft)

    @property
    def frequency(self):
        return np.linspace(0, self.freq_scale, len(self.fft))

    @staticmethod
    def fft(signal, window=windows.hann):
        """Apply fft with window function to signal, expect an equidistant
        timesteps and a signal with an average mean value of zero. Only
        return half spectrum.
        Use window = np.ones if no window function shall be applied.
        Refer to http://www.ni.com/white-paper/4278/en/ to choose a correct
        window"""
        # scales taken from http://www.ni.com/white-paper/4278/en/
        if window is windows.hann:
            window_scale = 1/0.5
        elif window is windows.hamming:
            window_scale = 1/0.54
        elif window is windows.blackman:
            window_scale = 1/0.42
        elif window is windows.blackmanharris:
            window_scale = 1/0.42
        elif window is windows.flattop:
            window_scale = 1/0.22
        elif window is np.ones:
            window_scale = 1
        else:
            window_scale = 1
            msg = 'Window {} not recognised, using scaling factor of 1'
            warnings.warn(msg.format(window))
        npoints = len(signal)
        fft = np.fft.fft(signal.vals*window(npoints))*1/2*window_scale
        half = int(np.ceil(npoints/2))
        return fft[:half]

    @staticmethod
    def equidistant(signal):
        """Takes a signal with varying time steps and returns it with
        equidistant timestamp. If all timessteps are already equal,
        the original signal is returned, if not, the signal is resampled
        where the new timestep is a half of the smallest original one"""
        dtimes = signal.dtimes
        if all(abs(dtime - dtimes[0]) <= 1e-12 for dtime in dtimes):
            return signal

        step = min(dtimes)
        t = signal.times
        y = signal.vals
        (yy, tt) = resample(y, t=t, num=int(np.ceil(t[-1]/step)))
        # TODO, shift signal to correct first time step

        return Signal(tt, yy)

    @staticmethod
    def fft_info(signal):
        """Expects equidistant time steps of signal, returns amplitude
        scaling and cut off freqency."""
        npoints = len(signal)
        ampl = 2/(np.ceil(npoints/2))
        step = signal.dtimes[-1]
        freq = 1/step/2
        amv = signal.amv
        return freq, ampl, amv

    def pplot(self, ax=None, plot=plt.plot, **kwargs):
        """Reasonable arguments for plot are: plt.plot, plt.semilogy,
        plt.semilogx, plt.loglog, where plt is matplotlib.pyplot"""
        ax = ax if ax else make_empty_axes()
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.thisplot = plot
        ax.thisplot(self.frequency, self.amplitude, **kwargs)


def nearest_power(n):
    return int(np.power(2, int(np.log2(n) + 0.5)))
