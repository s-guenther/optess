#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

import pyomo.environ as pe
from abc import ABC, abstractmethod

from optimstorage import Signal, Objective, NoResults, Solver


class DataIsNotCompletelyDefinedError(Exception):
    """Error class which is raised if pyomo model shall be constructed but
    not all model data is available"""
    pass


class AbstractOptimEES(ABC):
    """Returns an initalized or uninitialized optimmodel object
    which holds information about about a hybrid storage optimisation setting.
        signal      - load profile data (signal.t, signal.x values)
        base        - base storage data (base.power, base.efficiency,
                      base. discharge)
        peak        - as base
        objective   - 'power' or 'energy'
    When called, it builds a pyomo model object from the defined data.
    """
    def __init__(self, signal=None, objective=None,
                 solver='gurobi',
                 name='Hybrid Storage Optimization', info=None):
        self._signal = None

        self._objective = None
        self._strategy = None
        self._solver = None
        self._name = None

        self._model = None
        self._results = None

        if signal is not None:
            self.signal = signal
        if objective is not None:
            self.objective = objective

        self.solver = solver
        self.name = name

        # info is the only variable which is not typechecked and can be
        # freely modified. It can be used to store arbitrary userdata
        self.info = info


    # The following properties reset the model and results if set
    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, val):
        # noinspection PyArgumentList
        self._signal = Signal(val)
        self._modified()

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, val):
        # noinspection PyArgumentList
        self._objective = Objective(val)
        self._modified()

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, val):
        self._solver = Solver(val)
        self._modified()

    # name is only typechecked, but will not reset the results
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = str(val)

    # The following properties check if the variable is available, if not,
    # it is built; they are getonly
    @property
    def model(self):
        if self._model is None:
            self._build_pyomo_model()
        return self._model

    @property
    def results(self):
        if self._results is None:
            self._solve_pyomo_model()
        return self._results

    # Define convenience functions for output
    def pprint(self):
        """Pretty print the object state"""
        # TODO This should differ from __str__()
        # TODO Implement by delegating to composed objects
        # TODO Maybe prepare a hook for derived classes
        print(self.__str__())

    def pplot(self):
        """Pretty plot the object with matplotlib"""
        # TODO implement this stuff
        # TODO Implement by delegating to composed objects
        # TODO Maybe prepare a hook for derived classes
        pass

    def __repr__(self):
        """Return verbose string describing object"""
        # TODO implement
        self.__repr__()

    # Protected and private functions
    def _modified(self):
        """Automatically called if signal, base, peak, strategy or objective
        is changed. This resets model and results variable"""
        self._model = None
        self._results = None

    def _build_pyomo_model(self):
        """Create a Pyomo Model, save it internally"""
        if self._is_completely_defined():
            model = self._call_builder()
        else:
            raise DataIsNotCompletelyDefinedError()
        self._model = model

    def _solve_pyomo_model(self):
        """Solve the pyomo model, build it if neccessary, save internally"""
        # TODO separate object passed in init
        solver = pe.SolverFactory(self.solver.name)
        res = solver.solve(self.model)
        # TODO implement better validity checking
        valid = res['Solver'][0]['Status'].key == 'ok'
        if valid:
            self._results = self._call_results_generator()
        else:
            self._results = NoResults()

    @abstractmethod
    def _is_completely_defined(self):
        pass

    # GoF Strategy Pattern, delegate call to Subclasses
    @abstractmethod
    def _call_builder(self):
        pass

    # GoF Strategy Pattern, delegate call to Subclasses
    @abstractmethod
    def _call_results_generator(self):
        pass
