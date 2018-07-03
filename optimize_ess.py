#!/usr/bin/env python3
"""This module encapsulates model creation factory"""

import pyomo.environ as pe
from abc import ABC, abstractmethod

from hybridbuilder import HybridBuilder
from singlebuilder import SingleBuilder
from objective import Objective, Solver, Strategy
from results import SingleResults, HybridResults, NoResults
from powersignal import Signal
from storage import Storage
from utility import make_three_empty_axes


class DataIsNotCompletelyDefinedError(Exception):
    """Error class which is raised if pyomo model shall be constructed but
    not all model data is available"""
    pass


class NoFirstStageCalculatedError(Exception):
    """Error which is thrown in case the second stage shall be build but the
    first stage is not existent for now"""
    pass


class AbstractOptimizeESS(ABC):
    """Returns an initalized or uninitialized optimmodel object
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
                 name='Storage Optimization', info=None):
        self._signal = None

        self._objective = None
        self._solver = None
        self._name = None

        self._model = [None, None]
        self._results = [None, None]
        self._solverstatus = [None, None]

        if signal is not None:
            self.signal = signal
        if objective is not None:
            self.objective = objective

        self._builder = None
        self.solver = solver
        self.name = name

        # info is the only variable which is not typechecked and can be
        # freely modified. It can be used to store arbitrary userdata
        # TODO of any use? could be monkey patched at any time...
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
        if self._model[0] is None:
            self._build_1st_pyomo_model()
        return self._model

    @property
    def results(self):
        if self._results[0] is None:
            self._solve_pyomo_model()
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
        self._model = [None, None]
        self._results = [None, None]
        self._solverstatus = [None, None]

    def _build_1st_pyomo_model(self):
        """Create a Pyomo Model, save it internally"""
        if self._is_completely_defined():
            model_1st = self._build_1st_stage()
        else:
            raise DataIsNotCompletelyDefinedError()
        self._model[0] = model_1st

    def _build_2nd_pyomo_model(self):
        if self.model[0]:
            model_2nd = self._build_2nd_stage()
        else:
            raise NoFirstStageCalculatedError()
        self.model[1] = model_2nd

    def _solve_pyomo_model(self):
        """Solve the pyomo model, build it if neccessary, save internally"""
        # TODO separate object passed in init
        solver = pe.SolverFactory(self.solver.name)
        res = solver.solve(self.model[0])
        self._solverstatus[0] = res
        # TODO implement better validity checking
        valid = res['Solver'][0]['Status'].key == 'ok'
        if valid:
            self._results[0] = self._extract_results(self.model[0],
                                                     self.signal)
            # Build and calculate 2nd model
            self._build_2nd_pyomo_model()
            res_2nd = solver.solve(self.model[1])
            self._solverstatus[1] = res_2nd
            valid_2nd = res_2nd['Solver'][0]['Status'].key == 'ok'
            if valid_2nd:
                self._results[1] = self._extract_results(self.model[1],
                                                         self.signal)
            else:
                self._results[1] = NoResults()
        else:
            self._results = [NoResults()]*2

    @abstractmethod
    def _is_completely_defined(self):
        pass

    # GoF Strategy Pattern, delegate call to Subclasses
    @abstractmethod
    def _build_1st_stage(self):
        pass

    @abstractmethod
    def _build_2nd_stage(self):
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
                 solver='gurobi', name='Single Storage Optimization',
                 info=None):
        super().__init__(signal=signal, objective=objective, solver=solver,
                         name=name, info=info)

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

    @staticmethod
    def _extract_results(model, signal):
        return SingleResults(model, signal)

    def _build_1st_stage(self):
        return self._builder.build_1st_stage(self.signal, self.storage,
                                             self.objective)

    def _build_2nd_stage(self):
        return self._builder.build_2nd_stage()
