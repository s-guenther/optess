#!/usr/bin/env python3

from utility import make_empty_axes, make_two_empty_axes
from powersignal import Signal


class HybridResults:
    """Represents results of optimization in an easily accassible way"""
    def __init__(self, model, signal):
        """Writes results in pyomo model to optimstorage classes."""
        # Direct variables stored in model
        signalvarnames = ['base', 'peak',
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
                    Signal(signal.times,
                           getattr(model, varname).get_values().values()))

        for varname in floatvarnames:
            setattr(self,
                    varname,
                    list(getattr(model, varname).get_values().values())[0])

        # Derived variables
        self.both = self.base + self.peak
        self.bothinner = self.baseinner + self.peakinner

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

    def pprint(self, fig=100):
        # TODO implement
        print(self.__dict__)

    def pplot(self, ax=None):
        # TODO pass additional arguments to plot functions ?
        ax = ax if ax else make_empty_axes()

        # Define functions which extract pos/neg vals from a signal and a
        # function to apply these functions to a list of signals
        def get_pos_vals(signal):
            return ((signal >= 0)*signal).vals

        def get_neg_vals(signal):
            return ((signal < 0)*signal).vals

        def apply_fcn_to_signals(inputsignals, fcn):
            return [fcn(signal) for signal in inputsignals]

        # Calculate plotdata with helper functions
        signals = [self.baseinner, self.peakinner,
                   self.basesignedlosses, self.peaksignedlosses,
                   self.baseinter, self.peakinter]
        posvalvecs = apply_fcn_to_signals(signals, get_pos_vals)
        negvalvecs = apply_fcn_to_signals(signals, get_neg_vals)
        timevec = self.both.times

        # Plot positive and negative part of stackplot separately
        plotconfig = dict(step='pre',
                          colors=('#548b54', '#8b1a1a',  # palegreen, firebrick
                                  '#7ccd7c', '#cd2626',  # 4 - 3 - 1
                                  '#9aff9a', '#ff3030'))
        ax.stackplot(timevec, *posvalvecs, **plotconfig)
        ax.stackplot(timevec, *negvalvecs, **plotconfig)

        # add black zero line
        ax.axhline(color='black')
        # add both base/peak added
        self.both.pplot(ax=ax, color='blue', linewidth=2)

    def __repr__(self):
        strfmt = '<<{cls} at {resid}>, base={b}, peak={p}>'
        fields = dict(cls=self.__class__.__name__,
                      resid=hex(id(self)),
                      b=self.baseenergycapacity,
                      p=self.peakenergycapacity)
        return strfmt.format(**fields)


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
            setattr(self,
                    varname,
                    Signal(signal.times,
                           getattr(model, varname).get_values().values()))

        for varname in floatvarnames:
            setattr(self,
                    varname,
                    list(getattr(model, varname).get_values().values())[0])

        # Derived variables
        self.signedlosses = ((self.power - self.inner) * (self.power >= 0) +
                             (self.inner - self.power) * (self.power < 0))

    def pprint(self, fig=100):
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

        if ax2:
            self.energy.pplot(ax=ax2, color='blue', linewidth=2)

    def __repr__(self):
        strfmt = '<<{cls} at {resid}>, storage={s}>'
        fields = dict(cls=self.__class__.__name__,
                      resid=hex(id(self)),
                      s=self.energycapacity)
        return strfmt.format(**fields)


class NoResults:
    """Dummy Class which is returned if the solver failed"""
    def pplot(self, ax=None):
        pass

    def pprint(self):
        pass
