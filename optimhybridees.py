#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

from abstractoptimees import AbstractOptimEES
from hybridbuilder import HybridBuilder
from optimstorage import Storage, Strategy, HybridResults


class DataIsNotCompletelyDefinedError(Exception):
    """Error class which is raised if pyomo model shall be constructed but
    not all model data is available"""
    pass


class OptimHybridEES(AbstractOptimEES):
    """Returns an initalized or uninitialized optimmodel object
    which holds information about about a hybrid storage optimisation setting.
        signal      - load profile data (signal.t, signal.x vals)
        base        - base storage data (base.power, base.efficiency,
                      base. discharge)
        peak        - as base
        objective   - 'power' or 'energy'
    When called, it builds a pyomo model object from the defined data.
    """

    def __init__(self, signal=None, base=None, peak=None, objective=None,
                 strategy='inter', solver='gurobi',
                 name='Hybrid Storage Optimization', info=None):
        super().__init__(signal=signal, objective=objective, solver=solver,
                         name=name, info=info)

        self._base = None
        self._peak = None
        self._strategy = None

        if base is not None:
            self.base = base
        if peak is not None:
            self.peak = peak

        self.strategy = strategy

    # The following properties reset the model and results if set
    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, val):
        # noinspection PyArgumentList
        self._base = Storage(val)
        self._modified()

    @property
    def peak(self):
        return self._peak

    @peak.setter
    def peak(self, val):
        # noinspection PyArgumentList
        self._peak = Storage(val)
        self._modified()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, val):
        self._strategy = Strategy(val)
        self._modified()

    def _is_completely_defined(self):
        return all([self.signal, self.base, self.peak, self.objective])

    def _call_builder(self):
        return HybridBuilder.build(self.signal, self.base, self.peak,
                                   self.objective, self.strategy)

    def _call_results_generator(self):
        return HybridResults(self.model, self.signal)
