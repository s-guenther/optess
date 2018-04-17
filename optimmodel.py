#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

import pyomo.environ as pe

import optimmodelhelper as _helper
from optimstorage import Signal, Storage, Objective


class DataIsNotCompletelyDefinedError(Exception):
    """Error class which is raised if pyomo model shall be constructed but
    not all model data is available"""
    pass


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
        self.signal = Signal(signal)
        self.base = Storage(base)
        self.peak = Storage(peak)
        self.objective = Objective(objective.name)

    def __call__(self):
        """Create a Pyomo Model from the data if data is complete,
        else through an error message"""
        is_completely_defined = all([self.signal, self.base,
                                     self.peak, self.objective])
        if is_completely_defined:
            return self._make_pyomo_model()
        else:
            raise DataIsNotCompletelyDefinedError()

    def _make_pyomo_model(self):
        """Create a Pyomo Model"""
        model = pe.ConcreteModel()
        _helper.add_vars(self, model)
        _helper.add_constraints(self, model)
        _helper.add_objective(self, model)
        return model
