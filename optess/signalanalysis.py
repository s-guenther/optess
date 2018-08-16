#!/usr/bin/env python3
"""Functions and Methods which analyze a signal but are not included in the
signal class itself, as they are a little bit more complex."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import resample

from optess import Signal


class PEHMap:
    pass


# TODO pyplot.psd shows similar functionality...
class FFT:
    def __init__(self, signal):
        signal = self.equidistant(signal)
        (self.freq_scale, self.ampl_scale, self.amv) = self.fft_info(signal)
        self.fft = self.fft(signal - self.amv)

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
    def fft(signal, window=np.hanning):
        """Apply fft with window function to signal, expect an equidistant
        timesteps and a signal with an average mean value of zero. Only
        return half spectrum.
        Use window = np.ones if no window function shall be applied."""
        npoints = len(signal)
        fft = np.fft.fft(signal.vals*window(npoints))
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

    def pplot(self, plot=plt.plot):
        """Reasonable arguments for plot are: plt.plot, plt.semilogy,
        plt.semilogx, plt.loglog"""
        ax = plt.figure().add_subplot(1, 1, 1)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.thisplot = plot
        ax.thisplot(self.frequency, self.amplitude)
