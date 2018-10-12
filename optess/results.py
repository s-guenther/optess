#!/usr/bin/env python3

from collections import namedtuple
from random import randint
import os
import pickle
import copy

from .utility import make_two_empty_axes
from .signal import Signal
from .signalanalysis import PEHMap, PSD


# noinspection PyUnresolvedReferences
class HybridResults:
    """Represents results of optimization in an easily accessible way"""
    def __init__(self, model, signal):
        """Writes results in pyomo model to optimstorage classes."""
        # Direct variables stored in model
        signalvarnames = ['base', 'peak',
                          'baseinner', 'peakinner',
                          'inter',
                          'baseenergy', 'peakenergy']
        floatvarnames = ['baseenergycapacity', 'peakenergycapacity',
                         'baseenergyinit', 'peakenergyinit']

        for varname in signalvarnames:
            vals = list(getattr(model, varname).get_values().values())
            setattr(self, varname, Signal(signal.times, vals))
        for varname in floatvarnames:
            val = list(getattr(model, varname).get_values().values())[0]
            setattr(self, varname, val)

        # Derived variables
        self.power = self.base + self.peak
        self.powerinner = self.baseinner + self.peakinner

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

        self.basecycles, self.peakcycles = self._get_cycles()

    def pprint(self):
        # TODO implement
        print(self.__dict__)

    def pplot(self, ax=None):
        # TODO pass additional arguments to plot functions ?
        if ax is None:
            ax1, ax2 = make_two_empty_axes()
        else:
            try:
                ax1, ax2 = ax
            except TypeError:
                ax1 = ax
                ax2 = None

        # Define functions which extract pos/neg vals from a signal and a
        # function to apply these functions to a list of signals
        def get_pos_vals(signal):
            return ((signal >= 0)*signal).vals

        def get_neg_vals(signal):
            return ((signal < 0)*signal).vals

        def apply_fcn_to_signals(inputsignals, fcn):
            return [fcn(signal) for signal in inputsignals]

        # Calculate plotdata with helper functions
        baseinner_mod = (self.baseinner -
                         self.basesignedlosses*(self.basesignedlosses < 0))
        peakinner_mod = (self.peakinner -
                         self.peaksignedlosses*(self.peaksignedlosses < 0))
        signals = [baseinner_mod, peakinner_mod,
                   self.basesignedlosses, self.peaksignedlosses,
                   self.baseinter, self.peakinter]
        posvalvecs = apply_fcn_to_signals(signals, get_pos_vals)
        negvalvecs = apply_fcn_to_signals(signals, get_neg_vals)
        timevec = self.power.times

        # Plot positive and negative part of stackplot separately
        plotconfig = dict(step='pre',
                          colors=('#548b54', '#8b1a1a',  # palegreen, firebrick
                                  '#7ccd7c', '#cd2626',  # 4 - 3 - 1
                                  '#9aff9a', '#ff3030'))
        ax1.stackplot(timevec, *posvalvecs, **plotconfig)
        ax1.stackplot(timevec, *negvalvecs, **plotconfig)

        # add black zero line
        ax1.axhline(color='black')
        # add both base/peak added
        self.power.pplot(ax=ax1, color='blue', linewidth=2)

        ax1.autoscale(tight=True)

        if ax2:
            self.baseenergy.pplot(ax=ax2, color='#548b54', linewidth=2)
            self.peakenergy.pplot(ax=ax2, color='#8b1a1a', linewidth=2)
            ax2.autoscale(tight=True)

    def _get_cycles(self):
        """Returns base and peak cycles as tuple"""
        energybase = self.baseenergycapacity
        energypeak = self.peakenergycapacity
        basepower = self.base + self.baseinter
        peakpower = self.peak + self.peakinter
        return basepower.cycles(energybase), peakpower.cycles(energypeak)

    def __repr__(self):
        strfmt = '<<{cls} at {resid}>, base={b}, peak={p}>'
        fields = dict(cls=self.__class__.__name__,
                      resid=hex(id(self)),
                      b=self.baseenergycapacity,
                      p=self.peakenergycapacity)
        return strfmt.format(**fields)


# noinspection PyUnresolvedReferences
class SingleResults:
    """Represents results of optimization in an easily accassible way"""
    # noinspection PyArgumentList
    def __init__(self, model, signal):
        """Writes results in pyomo model to optimstorage classes."""
        # Direct variables stored in model
        signalvarnames = ['power',
                          'inner',
                          'energy']
        floatvarnames = ['energycapacity',
                         'energyinit']

        for varname in signalvarnames:
            vals = list(getattr(model, varname).get_values().values())
            setattr(self, varname, Signal(signal.times, vals))

        for varname in floatvarnames:
            val = list(getattr(model, varname).get_values().values())[0]
            setattr(self, varname, val)

        # Derived variables
        self.signedlosses = ((self.power - self.inner) * (self.power >= 0) +
                             (self.inner - self.power) * (self.power < 0))

        self.cycles = self._get_cycles()

    def pprint(self):
        # TODO implement
        print(self.__dict__)

    def pplot(self, ax=None):
        # TODO
        # it should not be neccessary to split pos and neg vals as signals
        # should always have the same sign at a specific point in time
        if ax is None:
            ax1, ax2 = make_two_empty_axes()
        else:
            try:
                ax1, ax2 = ax
            except TypeError:
                ax1 = ax
                ax2 = None

        timevec = self.power.times
        inner = (self.inner - self.signedlosses*(self.signedlosses < 0)
                 ).vals
        losses = self.signedlosses.vals
        plotconfig = dict(step='pre',
                          colors=('slateblue', 'cornflowerblue'))
        ax1.stackplot(timevec, inner, losses, **plotconfig)
        ax1.axhline(color='black')
        self.power.pplot(ax=ax1, color='blue', linewidth=2)

        ax1.autoscale(tight=True)

        if ax2:
            self.energy.pplot(ax=ax2, color='blue', linewidth=2)
            ax2.autoscale(tight=True)

    def __repr__(self):
        strfmt = '<<{cls} at {resid}>, storage={s}>'
        fields = dict(cls=self.__class__.__name__,
                      resid=hex(id(self)),
                      s=self.energycapacity)
        return strfmt.format(**fields)

    def _get_cycles(self):
        """Returns base and peak cycles as tuple"""
        return self.power.cycles(self.energycapacity)


class NoResults:
    """Dummy Class which is returned if the solver failed"""
    def pplot(self, ax=None):
        pass

    def pprint(self):
        pass


_Dim = namedtuple('_Dim', 'power energy')
_Norm = namedtuple('_Norm', 'power energy')
_SignalParameters = namedtuple('_SignalParameters', 'arv rms form crest')


class ReducedHybridResults:
    """Writes Results to Disc and only holds relevant integral variables"""
    def __init__(self, optim_case, savepath='', save_to_disc=True):
        """Takes an OptimizeHybridESS object and extracts data"""
        results = optim_case.results

        self.basedim = _Dim(optim_case.base.power, results.baseenergycapacity)
        self.peakdim = _Dim(optim_case.peak.power, results.peakenergycapacity)

        single_power_min = self.basedim.power.min + self.peakdim.power.min
        single_power_max = self.basedim.power.max + self.peakdim.power.max
        single_energy = self.basedim.energy + self.peakdim.energy

        self.basenorm = _Norm((self.basedim.power.min/single_power_min,
                               self.basedim.power.max/single_power_max),
                              self.basedim.energy/single_energy)
        self.peaknorm = _Norm((self.peakdim.power.min/single_power_min,
                               self.peakdim.power.max/single_power_max),
                              self.peakdim.energy/single_energy)

        self.baselosses = max(abs(results.basesignedlosses).integrate().vals)
        self.peaklosses = max(abs(results.peaksignedlosses).integrate().vals)

        self.baseparameters = _SignalParameters(results.baseinner.arv,
                                                results.baseinner.rms,
                                                results.baseinner.form,
                                                results.baseinner.crest)
        self.peakparameters = _SignalParameters(results.peakinner.arv,
                                                results.peakinner.rms,
                                                results.peakinner.form,
                                                results.peakinner.crest)

        self.basecycles = results.basecycles
        self.peakcycles = results.peakcycles
        self.basepeh = PEHMap(results.base, results.baseenergy)
        self.peakpeh = PEHMap(results.peak, results.peakenergy)
        self.basepsd = PSD(results.base)
        self.peakpsd = PSD(results.peak)
        self.chargebase = max((results.inter *
                               (results.inter >= 0)).integrate().vals)
        self.chargepeak = -max((results.inter *
                                (results.inter <= 0)).integrate().vals)

        self._filename = None
        if save_to_disc:
            self._save_to_disc(savepath, optim_case)

    def _save_to_disc(self, savepath, optim_case):
        tag = '{:04x}'.format(randint(0, 16**4))
        while os.path.isfile(os.path.join(savepath, tag)):
            tag = '{:04x}'.format(randint(0, 16**4))
        self._filename = os.path.join(savepath, tag)
        optim_case.save(self._filename)

    # TODO move load and save functions to separate library
    def load_all_results(self):
        filename, fileend = self._filename, 'opt'
        sep = '.'
        with open(sep.join([filename, fileend]), 'rb') as file:
            opt_case = pickle.load(file)
        return opt_case


class ReducedSingleResults:
    """Writes Results to Disc and only holds relevant integral variables"""
    def __init__(self, optim_case, savepath='', save_to_disc=True):
        """Takes an OptimizeHybridESS object and extracts data"""
        results = optim_case.results

        self.dim = _Dim(optim_case.storage.power, results.energycapacity)

        self.losses = max(abs(results.signedlosses).integrate().vals)

        self.signalparameters = _SignalParameters(results.inner.arv,
                                                  results.inner.rms,
                                                  results.inner.form,
                                                  results.inner.crest)
        self.cycles = results.cycles
        self.peh = PEHMap(results.power, results.energy)
        self.psd = PSD(results.power)

        self._filename = None
        if save_to_disc:
            self._save_to_disc(savepath, optim_case)

        # TODO add data which is needed for plotting
        # add all, but resample to a reasonable amount of data

    def _save_to_disc(self, savepath, optim_case):
        tag = '{:04x}'.format(randint(0, 16**4))
        while os.path.isfile(os.path.join(savepath, tag)):
            tag = '{:04x}'.format(randint(0, 16**4))
        self._filename = os.path.join(savepath, tag)
        optim_case.save(self._filename)

    # TODO move load and save functions to separate library
    def load_all_results(self):
        """Load an optimize ess object"""
        filename, fileend = self._filename, 'opt'
        sep = '.'
        with open(sep.join([filename, fileend]), 'rb') as file:
            opt_case = pickle.load(file)
        return opt_case


# TODO these functions only transform reduced results but not complete results
def single_to_base_results(singlered):
    hybridred = copy.copy(singlered)
    hybridred.__class__ = ReducedHybridResults

    hybridred.basedim = hybridred.dim
    hybridred.peakdim = _Dim(0, 0)
    hybridred.basenorm = _Norm(1, 1)
    hybridred.peaknorm = _Norm(0, 0)

    hybridred.baselosses = hybridred.losses
    hybridred.peaklosses = 0

    hybridred.baseparameters = hybridred.signalparameters
    hybridred.peakparameters = _SignalParameters(0, 0, 0, 0)

    hybridred.basecycles = hybridred.cycles
    hybridred.peakcycles = 0

    hybridred.basepeh = hybridred.peh
    hybridred.peakpeh = copy.deepcopy(hybridred.peh)
    hybridred.peakpeh.eweights *= 0
    hybridred.peakpeh.pweights *= 0
    hybridred.peakpeh.map *= 0

    hybridred.basepsd = hybridred.psd
    hybridred.peakpsd = copy.deepcopy(hybridred.psd)
    hybridred.peakpsd.amv = 0
    hybridred.peakpsd.psd *= 0

    hybridred.chargepeak = 0
    hybridred.chargebase = 0

    return hybridred


def single_to_peak_results(singlered):
    hybridred = copy.copy(singlered)
    hybridred.__class__ = ReducedHybridResults

    hybridred.peakdim = hybridred.dim
    hybridred.basedim = _Dim(0, 0)
    hybridred.peaknorm = _Norm(1, 1)
    hybridred.basenorm = _Norm(0, 0)

    hybridred.peaklosses = hybridred.losses
    hybridred.baselosses = 0

    hybridred.peakparameters = hybridred.signalparameters
    hybridred.baseparameters = _SignalParameters(0, 0, 0, 0)

    hybridred.peakcycles = hybridred.cycles
    hybridred.basecycles = 1

    hybridred.peakpeh = hybridred.peh
    hybridred.basepeh = copy.deepcopy(hybridred.peh)
    hybridred.basepeh.eweights *= 0
    hybridred.basepeh.pweights *= 0
    hybridred.basepeh.map *= 0

    hybridred.peakpsd = hybridred.psd
    hybridred.basepsd = copy.deepcopy(hybridred.psd)
    hybridred.basepsd.amv = 0
    hybridred.basepsd.psd *= 0

    hybridred.chargepeak = 0
    hybridred.chargebase = 0
    return hybridred

# TODO refactor Results and Reduced Results in a way that they are identical
# TODO for Single and Hybrid --> Storage Results --> Hybrid takes two of them
