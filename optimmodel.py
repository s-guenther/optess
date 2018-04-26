#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

import pyomo.environ as pe

import buildmodel as _buildmodel
from optimstorage import Signal, Storage, Objective, Strategy


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
    def __init__(self, signal=None, base=None, peak=None, objective=None,
                 strategy=None, name='Hybrid Storage Optimization'):
        self.signal = Signal(signal)
        self.base = Storage(base)
        self.peak = Storage(peak)
        self.objective = Objective(objective.name)
        self.strategy = Strategy(strategy.name)
        self.name = str(name)
        self._model = None
        self._results = None

    def __call__(self):
        """Create a Pyomo Model from the data if data is complete,
        else through an error message"""
        is_completely_defined = all([self.signal, self.base,
                                     self.peak, self.objective])
        if is_completely_defined:
            return self._pyomo_model()
        else:
            raise DataIsNotCompletelyDefinedError()

    def _pyomo_model(self):
        """Create a Pyomo Model"""
        model = pe.ConcreteModel(name=self.name)
        self._add_vars(model)
        self._add_constraints(model)
        self._add_strategy(model)
        self._add_objective(model)
        return model

    def solve(self):
        pass

    def results(self):
        pass

    def pprint(self):
        pass

    def pplot(self):
        pass

    def _add_strategy(self, model):
        if self.strategy == Strategy.inter:
            _buildmodel.add_inter_constraint()
        elif self.strategy == Strategy.nointer:
            _buildmodel.add_nointer_constraint(model)

    def _add_vars(self, model):
        _buildmodel.add_vars(self, model)

    def _add_constraints(self, model):
        _buildmodel.add_constraints(self, model)

    def _add_objective(self, model):

        if self.objective == Objective.power:
            _buildmodel.add_peak_cutting_objective(self, model)
        elif self.objective == Objective.energy:
            _buildmodel.add_throughput_objective(self, model)

        _buildmodel.add_capacity_minimizing_objective(self, model)
