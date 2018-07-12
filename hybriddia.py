#!/usr/bin/env python3

import itertools
import scipy.interpolate as interp
from numpy import linspace
import multiprocessing as mp


from optimize_ess import OptimizeSingleESS, OptimizeHybridESS
from powersignal import Signal
from storage import Storage
from objective import Objective, Solver
from matplotlib import pyplot as plt
from collections import defaultdict


class PointOutsideAreaError(ValueError):
    """Raised if a point within the Hybridisation Area shall be calculated,
    but the point is outside"""
    pass


class OnTheFlyDict(defaultdict):
    """Performs for a specific cut which is missing in the inter/nointer
    dict of HybridDia the adequate calculation and adds it the dict"""
    def __init__(self, hybdia, strategy):
        super().__init__()
        self.hybdia = hybdia
        self.strategy = strategy

    def __missing__(self, key):
        optim_case = self.hybdia.calculate_cut(key, self.strategy, True)
        return optim_case


class HybridDia:
    # noinspection PyArgumentList
    def __init__(self, signal, singlestorage, objective, solver='gurobi',
                 name='Hybridisation Diagram'):
        self.signal = Signal(signal)
        self.storage = Storage(singlestorage)
        self.objective = Objective(objective)
        self.solver = Solver(solver)
        self.single = self.calculate_single()
        # self.inter = OnTheFlyDict(self, 'inter')
        # self.nointer = OnTheFlyDict(self, 'nointer')
        self.inter = dict()
        self.nointer = dict()
        self.area = dict()
        self.name = str(name)

        self.powercapacity = self.storage.power.max
        self.energycapacity = self.single.results.energycapacity

    def calculate_single(self):
        single = OptimizeSingleESS(self.signal, self.storage,
                                   self.objective, self.solver)
        return single

    def calculate_cut(self, cut, strategy='inter', add_to_internal_list=True):
        signal = self.signal
        base = cut*self.storage
        peak = (1 - cut)*self.storage
        objective = self.objective
        solver = self.solver

        optim_case = OptimizeHybridESS(signal, base, peak, objective,
                                       strategy, solver)
        optim_case.solve_pyomo_model()

        if add_to_internal_list:
            if strategy == 'inter':
                self.inter[cut] = optim_case
            elif strategy == 'nointer':
                self.nointer[cut] = optim_case
        return optim_case

    def calculate_both(self, cut):
        print('Starting cut={}'.format(cut))
        inter = self.calculate_cut(cut, 'inter',
                                   add_to_internal_list=False)
        nointer = self.calculate_cut(cut, 'nointer',
                                     add_to_internal_list=False)
        print('Ending cut={}'.format(cut))
        return cut, inter, nointer

    def calculate_curves(self, cuts=(0.2, 0.4, 0.5, 0.6, 0.8)):
        with mp.Pool() as pool:
            res = pool.map(self.calculate_both, cuts)

        for cut, inter, nointer in res:
            self.inter[cut] = inter
            self.nointer[cut] = nointer

    def calculate_area(self, raster=(7, 7)):
        """Calculates points within the hybridisation area to determine the
        cycle map, raster describes the number of number of points that will
        be generated in power and energy direction"""

        if not self.inter:
            self.calculate_curves()

        powercap = self.powercapacity
        energycap = self.energycapacity

        def raster_vector(npoints):
            return linspace(1/(npoints+2), (npoints+1)/(npoints+2), npoints)

        powers = raster_vector(raster[0])*powercap
        energies = raster_vector(raster[1])*energycap

        # TODO parallelize this code
        for point in itertools.product(energies, powers):
            if self.is_point_in_area(*point):
                print('Calculating area point {}'.format(point))
                self.calculate_point(*point)

    def is_point_in_area(self, energy, power):
        """Return True if a point is within the hybridisation area."""
        if not self.inter:
            self.calculate_curves()

        powercap = self.powercapacity
        cut = power/powercap
        if cut < 0 or cut > 1:
            return False

        # Interpolate Hybridisation Curve with given points
        hcuts = sorted(self.inter.keys())
        henergies = [self.inter[hcut].results.baseenergycapacity
                     for hcut in hcuts]
        hcurve = interp.interp1d(hcuts, henergies, 'linear')

        minenergy = self.energycapacity*cut  # left side of area at spec. cut
        maxenergy = hcurve(cut)  # right side of area at specified cut

        return minenergy <= energy <= maxenergy

    def calculate_point(self, energy, power):
        """The optimisation problem is solved for this point defined by
        power and energy."""
        cut = power/self.powercapacity
        base = cut*self.storage
        peak = (1-cut)*self.storage
        optim_case = OptimizeHybridESS(self.signal, base, peak,
                                       self.objective, solver=self.solver)

        baseenergy = energy
        peakenergy = self.energycapacity - energy
        optim_case.solve_pyomo_model(baseenergy, peakenergy)

        self.area[(power, energy)] = optim_case

    def pprint(self):
        # TODO implement
        pass

    def pplot(self):
        if not self.inter or not self.nointer:
            self.calculate_curves()
        # if not self.area:
        #     self.calculate_area()

        # TODO remove duplicate code, refactor
        powerinter = []
        inter = []
        cyclesinter = []
        for cut in sorted(self.inter.keys()):
            powerinter.append(cut*self.powercapacity)
            optim_case = self.inter[cut]
            inter.append(optim_case.results.baseenergycapacity)
            cyclesinter.append(self._get_cycles(optim_case))

        powernointer = []
        nointer = []
        cyclesnointer = []
        for cut in sorted(self.nointer.keys()):
            powernointer.append(cut*self.powercapacity)
            optim_case = self.nointer[cut]
            nointer.append(optim_case.results.baseenergycapacity)
            cyclesnointer.append(self._get_cycles(optim_case))

        ax = plt.figure().add_subplot(1, 1, 1)
        ax.plot([0, self.energycapacity], [0, self.powercapacity])
        ax.plot(nointer, powernointer, linestyle='--')
        ax.plot(inter, powerinter)

        for x, y, cycles in zip(nointer, powernointer, cyclesnointer):
            ax.text(x, y, '{:.2f}, {:.2f}'.format(*cycles),
                    HorizontalAlignment='right', VerticalAlignment='bottom')
        for x, y, cycles in zip(inter, powerinter, cyclesinter):
            ax.text(x, y, '{:.2f}, {:.2f}'.format(*cycles),
                    HorizontalAlignment='left', VerticalAlignment='top')
        for (y, x), opt_case in self.area.items():
            cbase, cpeak = self._get_cycles(opt_case)
            ax.text(x, y, '{:.2f}'.format(cbase),
                    color='blue', VerticalAlignment='top')
            ax.text(x, y, '{:.2f}'.format(cpeak),
                    color='red', VerticalAlignment='bottom')

        ax.set_ylabel('Power')
        ax.set_xlabel('Energy')
        ax.autoscale(tight=True)

    @staticmethod
    def _get_cycles(optim_case):
        """Returns base and peak cycles as tuple"""
        energybase = optim_case.results.baseenergycapacity
        energypeak = optim_case.results.peakenergycapacity
        basepower = optim_case.results.base + optim_case.results.baseinter
        peakpower = optim_case.results.peak + optim_case.results.peakinter
        return basepower.cycles(energybase), peakpower.cycles(energypeak)

# TODO __repr__
