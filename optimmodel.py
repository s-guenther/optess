#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

import pyomo.environ as pe

import build_hybrid_model as _buildmodel
from optimstorage import Signal, Storage, Objective, Strategy, Results, \
    NoResults, Solver


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
                 strategy='inter', solver='gurobi',
                 name='Hybrid Storage Optimization', info=None):
        self._signal = None
        self._base = None
        self._peak = None

        self._objective = None
        self._strategy = None
        self._solver = None
        self._name = None

        self._model = None
        self._results = None

        if signal is not None:
            self.signal = signal
        if base is not None:
            self.base = base
        if peak is not None:
            self.peak = peak
        if objective is not None:
            self.objective = objective

        self.strategy = strategy
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
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, val):
        # noinspection PyArgumentList
        self._objective = Objective(val)
        self._modified()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, val):
        self._strategy = Strategy(val)
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
        print(self.__str__())

    def pplot(self):
        """Pretty plot the object with matplotlib"""
        # TODO implement this stuff
        pass

    # Protected and private functions
    def _modified(self):
        """Automatically called if signal, base, peak, strategy or objective
        is changed. This resets model and results variable"""
        self._model = None
        self._results = None

    def _build_pyomo_model(self):
        """Create a Pyomo Model, save it internally"""
        is_completely_defined = all([self.signal, self.base,
                                     self.peak, self.objective])
        if is_completely_defined:
            model = pe.ConcreteModel(name=self.name)
            self._add_vars(model)
            self._add_constraints(model)
            self._add_strategy(model)
            self._add_objective(model)
        else:
            raise DataIsNotCompletelyDefinedError()
        self._model = model

    def _solve_pyomo_model(self):
        """Solve the pyomo model, build it if neccessary, save internally"""
        solver = pe.SolverFactory(self.solver.name)
        res = solver.solve(self.model)
        # TODO implement better validity checking
        valid = res['Solver'][0]['Status'].key == 'ok'
        if valid:
            self._results = Results(self.model, self.signal)
        else:
            self._results = NoResults()

    # Load strategy building functions from module buildmodel into class
    def _add_strategy(self, model):
        """Add strategy constraints to pyomo model"""
        if self.strategy.name == 'inter':
            _buildmodel.add_inter_constraint()
        elif self.strategy.name == 'nointer':
            _buildmodel.add_nointer_constraint(model)

    # Load variable building functions from module buildmodel into class
    def _add_vars(self, model):
        """Add variable definitions to pyomo model"""
        _buildmodel.add_vars(self, model)

    # Load contraint building functions from module buildmodel into class
    def _add_constraints(self, model):
        """Add constraints to pyomo model"""
        _buildmodel.add_constraints(self, model)

    # Load objective building functions from module buildmodel into class
    def _add_objective(self, model):
        """Adds objective (in a larger sense) or aim to pyomo model"""
        if self.objective.type == 'power':
            _buildmodel.add_peak_cutting_objective(self, model)
        elif self.objective.type == 'energy':
            _buildmodel.add_throughput_objective(self, model)

        _buildmodel.add_capacity_minimizing_objective(model)

    def __str__(self):
        """Return verbose string describing object"""
        # TODO implement
        self.__repr__()
