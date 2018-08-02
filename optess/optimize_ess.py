#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

import pyomo.environ as pe
from abc import ABC, abstractmethod
import copy
import pickle

from .hybridbuilder import HybridBuilder
from .singlebuilder import SingleBuilder
from .objective import Objective, Solver, Strategy
from .results import SingleResults, HybridResults, NoResults
from .signal import Signal
from .storage import Storage
from .utility import make_three_empty_axes


class DataIsNotCompletelyDefinedError(Exception):
    """Error class which is raised if pyomo model shall be constructed but
    not all model data is available"""
    pass


class NoFirstStageCalculatedError(Exception):
    """Error which is thrown in case the second stage shall be build but the
    first stage is not existent for now"""
    pass


class AbstractOptimizeESS(ABC):
    """Returns an initialized or uninitialized optimmodel object
    which holds information about a hybrid storage optimisation setting.
        signal      - load profile data (signal.t, signal.x values)
        base        - base storage data (base.power, base.efficiency,
                      base. discharge)
        peak        - as base
        objective   - 'power' or 'energy'
    When called, it builds a pyomo model object from the defined data.
    """
    def __init__(self, signal=None, objective=None,
                 solver='gurobi',
                 name='Storage Optimization'):
        self._signal = None

        self._objective = None
        self._solver = None
        self._name = None

        self._model = None
        self._results = None
        self._solverstatus = None

        if signal is not None:
            self.signal = signal
        if objective is not None:
            self.objective = objective

        self.solver = solver
        self.name = name

        self._builder = None

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
            self._model = self._build()
        return self._model

    @property
    def results(self):
        if self._results is None:
            self.solve_pyomo_model()
        return self._results

    # Define convenience functions for output
    def pprint(self):
        """Pretty print the object state"""
        # TODO This should differ from __str__()
        # TODO Implement by delegating to composed objects
        # TODO Maybe prepare a hook for derived classes
        print(self.__str__())

    def pplot(self, ax=None):
        """Pretty plot the object with matplotlib"""
        if ax is None:
            ax1, ax2, ax3 = make_three_empty_axes()
        else:
            try:
                ax1, ax2, ax3 = ax
            except TypeError:
                ax1 = ax
                ax2, ax3 = (None, None)
            except ValueError:
                ax1 = ax[0]
                ax2 = ax[1]
                ax3 = None

        original_signal = self.signal
        new_signal = self.signal + self.results.power

        original_signal.pplot(ax=ax1, color='grey', linewidth=2)
        new_signal.pplot(ax=ax1, color='black', linewidth=3)
        # noinspection PyUnresolvedReferences
        bool_ge = (original_signal > new_signal).vals
        bool_ge = [bool(val) for val in bool_ge]
        # this compensates a bug in fill_between to not draw the first element
        for ii in range(1, len(bool_ge)):
            if bool_ge[ii] and not bool_ge[ii-1]:
                bool_ge[ii-1] = True
        # noinspection PyUnresolvedReferences

        bool_le = (original_signal < new_signal).vals
        bool_le = [bool(val) for val in bool_le]
        # this compensates a bug in fill_between to not draw the first element
        for ii in range(1, len(bool_le)):
            if bool_le[ii] and not bool_le[ii-1]:
                bool_le[ii-1] = True
        ax1.fill_between(original_signal.times,
                         original_signal.vals, new_signal.vals,
                         where=bool_ge, step='pre', color='LightSkyBlue')
        ax1.fill_between(original_signal.times,
                         original_signal.vals, new_signal.vals,
                         where=bool_le, step='pre', color='LightSalmon')
        ax1.autoscale(tight=True)
        ax1.set_ylabel('Input Signal')

        if ax2 and ax3:
            self.results.pplot(ax=(ax2, ax3))
        elif ax2 and not ax3:
            self.results.pplot(ax=ax2)

    # def __repr__(self):
    #     """Return verbose string describing object"""
    #     # TODO implement
    #     pass

    # Protected and private functions
    def _modified(self):
        """Automatically called if signal, base, peak, strategy or objective
        is changed. This resets model and results variable"""
        self._model = None
        self._results = None
        self._solverstatus = None

    def _build_pyomo_model(self):
        """Create a Pyomo Model, save it internally"""
        if self._is_completely_defined():
            model = self._build()
        else:
            raise DataIsNotCompletelyDefinedError()
        self._model = model

    def save(self, filename):
        """Saves the object to disc"""
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'opt'

        with open(sep.join([filename, fileend]), 'wb') as file:
            pickle.dump(self, file)

    # TODO move load and save functions to separate library
    @staticmethod
    def load(filename):
        """Load an optimize ess object"""
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'opt'

        with open(sep.join([filename, fileend]), 'rb') as file:
            opt_case = pickle.load(file)

        return opt_case

    @abstractmethod
    def solve_pyomo_model(self):
        pass

    @abstractmethod
    def _is_completely_defined(self):
        pass

    # GoF Strategy Pattern, delegate call to Subclasses
    @abstractmethod
    def _build(self):
        pass

    # GoF Strategy Pattern, delegate call to Subclasses
    @staticmethod
    @abstractmethod
    def _extract_results(model, signal):
        pass


class OptimizeHybridESS(AbstractOptimizeESS):
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
                 name='Hybrid Storage Optimization'):
        super().__init__(signal=signal, objective=objective, solver=solver,
                         name=name)

        self._base = None
        self._peak = None
        self._strategy = None

        if base is not None:
            self.base = base
        if peak is not None:
            self.peak = peak

        self.strategy = strategy

        self._builder = HybridBuilder()
        self._first_stage = None

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

    def solve_pyomo_model(self, baseenergy=None, peakenergy=None):
        solver = pe.SolverFactory(self.solver.name)
        # If no dimensions are provided, then solve the model for the first
        # time and extract dimensions from the solution
        if baseenergy is None or peakenergy is None:
            self._solverstatus = solver.solve(self.model)
            valid = self._solverstatus['Solver'][0]['Status'].key == 'ok'
            if valid:
                self._results = self._extract_results(self.model, self.signal)
            else:
                self._results = NoResults()
                return
            first_stage = copy.copy(self)
            self._first_stage = first_stage
            baseenergy = self.results.baseenergycapacity
            peakenergy = self.results.peakenergycapacity
        # Then, build and solve second stage
        self._build_2nd_pyomo_model(baseenergy, peakenergy)
        self._solverstatus = solver.solve(self.model)
        valid = self._solverstatus['Solver'][0]['Status'].key == 'ok'
        if valid:
            self._results = self._extract_results(self.model, self.signal)
        else:
            self._results = NoResults()

    def _is_completely_defined(self):
        return all([self.signal, self.base, self.peak, self.objective])

    def _modified(self):
        super()._modified()
        self._first_stage = None

    def _build(self):
        return self._builder.minimize_energy(self.signal, self.base, self.peak,
                                             self.objective, self.strategy)

    def _build_2nd_pyomo_model(self, baseenergy, peakenergy):
        if self.model:
            model_2nd = self._builder.minimize_cycles(baseenergy, peakenergy)
        else:
            raise NoFirstStageCalculatedError()
        self._model = model_2nd

    def _build_2nd_stage(self):
        return self._builder.minimize_cycles()

    @staticmethod
    def _extract_results(model, signal):
        return HybridResults(model, signal)


class OptimizeSingleESS(AbstractOptimizeESS):
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
                 solver='gurobi', name='Single Storage Optimization'):
        super().__init__(signal=signal, objective=objective, solver=solver,
                         name=name)

        self._storage = None
        self._builder = SingleBuilder()

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

    def solve_pyomo_model(self):
        """Solve the pyomo model, build it if neccessary, save internally"""
        solver = pe.SolverFactory(self.solver.name)
        model = self.model
        res = solver.solve(model)
        self._solverstatus = res
        # TODO implement better validity checking
        valid = res['Solver'][0]['Status'].key == 'ok'
        if valid:
            self._results = self._extract_results(self.model,
                                                  self.signal)
        else:
            self._results = NoResults()

    @staticmethod
    def _extract_results(model, signal):
        return SingleResults(model, signal)

    def _build(self):
        return self._builder.minimize_energy(self.signal, self.storage,
                                             self.objective)
