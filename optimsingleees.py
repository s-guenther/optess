#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

from abstractoptimees import AbstractOptimEES
from singlebuilder import SingleBuilder
from optimstorage import Storage, SingleResults


class DataIsNotCompletelyDefinedError(Exception):
    """Error class which is raised if pyomo model shall be constructed but
    not all model data is available"""
    pass


class OptimSingleEES(AbstractOptimEES):
    """Returns an initalized or uninitialized optimmodel object
    which holds information about about a hybrid storage optimisation setting.
        signal      - load profile data (signal.t, signal.x vals)
        base        - base storage data (base.power, base.efficiency,
                      base. discharge)
        peak        - as base
        objective   - 'power' or 'energy'
    When called, it builds a pyomo model object from the defined data.
    """

    def __init__(self, signal=None, storage=None, objective=None,
                 solver='gurobi', name='Hybrid Storage Optimization',
                 info=None):
        super().__init__(signal=signal, objective=objective, solver=solver,
                         name=name, info=info)

        self._storage = None

        if storage is not None:
            self.storage = storage

    # The following properties reset the model and results if set
    @property
    def storage(self):
        return self._storage

    @storage.setter
    def storage(self, val):
        # noinspection PyArgumentList
        self._storage = Storage(val)
        self._modified()

    def _is_completely_defined(self):
        return all([self.signal, self.storage, self.objective])

    def _call_builder(self):
        return SingleBuilder.build(self.signal, self.storage, self.objective)

    def _call_results_generator(self):
        return SingleResults(self.model, self.signal)
