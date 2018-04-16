#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

from optimhelper import Signal, Storage


class OptimModel:
    """Returns an initalized or uninitialized optimmodel object
    which holds information about about a hybrid storage optimisation setting.
        signal      - load profile data (signal.t, signal.x values)
        base        - base storage data (base.power, base.efficiency,
                      base. discharge)
        peak        - as base
        objective   - 'power' or 'energy'
    When called, it builds a pyomo model object from the defined data.
    """
    def __init__(self, signal=None, base=None, peak=None, objective=None):
        self.signal = signal
        self.base = base
        self.peak = peak
        self.objective = objective

        # protected property variables
        self._signal = None
        self._base = None
        self._peak = None
        self._objective = None

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, val):
        if isinstance(val, Signal):
            self._signal = val
        else:
            self._signal = Signal(*val)

    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, val):
        if isinstance(val, Storage):
            self._base = val
        else:
            self._base = Storage(*val)

    @property
    def peak(self):
        return self._peak

    @peak.setter
    def peak(self, val):
        if isinstance(val, Storage):
            self._peak = val
        else:
            self._peak = Storage(*val)

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, val):
        try:
            if val.lower() is 'energy' or val.lower() is 'power':
                self._objective = val.lower()
            else:
                raise ValueError("Objective must be 'energy' or 'power'")
        except AttributeError:
            TypeError('Objective must be string')
