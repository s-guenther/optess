#!/usr/bin/env python3

from collections import OrderedDict

from .signal import Signal
from .objective import Objective, Solver, Strategy
from .storage import FullStorage


class UnknownPlotTypeError(ValueError):
    pass


class UnknownComputeTypeError(ValueError):
    pass


class OverdimDia:
    # noinspection PyArgumentList
    def __init__(self, signal, reduced_hybrid_result, objective,
                 strategy='inter', solver='gurobi', name='Overdimensioning'):
        self.signal = Signal(signal)
        self.red_hyb_result = reduced_hybrid_result
        self.objective = Objective(objective)
        self.solver = Solver(solver)
        self.strategy = Strategy(strategy)
        self.name = str(name)

        # Build Full Storages

        # Result Fields
        self.base = OrderedDict()
        self.peak = OrderedDict()
        self.energy = OrderedDict()
        self.power = OrderedDict()
        self.cross_ebpp = OrderedDict()
        self.cross_eppb = OrderedDict()
        self.dim = OrderedDict()
        self.mixed = OrderedDict()

    # --- Calculation routines
    def compute_base(self):
        pass

    def compute_peak(self):
        pass

    def compute_energy(self):
        pass

    def compute_power(self):
        pass

    def compute_cross_ebpp(self):
        pass

    def compute_cross_eppb(self):
        pass

    def compute_dim(self):
        pass

    def compute_mixed(self):
        pass

    def compute(self, ctype='mixed'):
        compute_fcn_name = 'compute_' + ctype
        try:
            compute_fcn = getattr(self, compute_fcn_name)
        except AttributeError:
            raise UnknownComputeTypeError
        compute_fcn()

    def _abstract_compute(self):
        pass

    # --- Plotting Routines
    def pplot_base(self, ax=None):
        pass

    def pplot_peak(self, ax=None):
        pass

    def pplot_energy(self, ax=None):
        pass

    def pplot_power(self, ax=None):
        pass

    def pplot_cross_ebpp(self, ax=None):
        pass

    def pplot_cross_eppb(self, ax=None):
        pass

    def pplot_dim(self, ax=None):
        pass

    def pplot_mixed(self, ax=None):
        pass

    def pplot(self, ax=None, ptype='mixed'):
        pplot_fcn_name = 'pplot_' + ptype
        try:
            pplot_fcn = getattr(self, pplot_fcn_name)
        except AttributeError:
            raise UnknownPlotTypeError
        pplot_fcn(ax=ax)

    def _abstract_pplot(self, ax=None):
        pass
