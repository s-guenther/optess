#!/usr/bin/env python3

import itertools
import scipy.interpolate as interp
from numpy import linspace

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
        self.inter = OnTheFlyDict(self, 'inter')
        self.nointer = OnTheFlyDict(self, 'nointer')
        self.area = dict()
        self.name = str(name)

        self.powercapacity = self.single.results.powercapacity
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

    def calculate_curves(self, cuts=(0.1, 0.25, 0.4, 0.6, 0.9)):

        for cut in cuts:
            # TODO parallelize this code
            print('Starting cut={}'.format(cut))
            self.calculate_cut(cut, 'inter')
            self.calculate_cut(cut, 'nointer')
            print('Ending cut={}'.format(cut))

    def calculate_area(self, raster=(7, 7)):
        """Calculates points within the hybridisation area to determine the
        cycle map, raster describes the number of number of points that will
        be generated in power and energy direction"""

        if not self.inter:
            self.calculate_curves()

        powercap = self.powercapacity
        energycap = self.energycapacity

        def raster_vector(npoints):
            return linspace(1/(npoints+2), (npoints+1)/npoints+2, npoints)

        powers = raster_vector(raster[0])*powercap
        energies = raster_vector(raster[1])*energycap

        # TODO parallelize this code
        for point in itertools.product(powers, energies):
            if self.is_point_in_area(*point):
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
        hcuts = [sorted(self.inter.keys())]
        henergies = [self.inter[hcut] for hcut in hcuts]
        hcurve = interp.interp1d(hcuts, henergies, 'linear')

        minenergy = energy*cut  # left side of area at specified cut
        maxenergy = hcurve(cut)  # right side of area at specified cut

        return minenergy < energy < maxenergy

    def calculate_point(self, power, energy):
        """The optimisation problem is solved for this point defined by
        power and energy."""
        # TODO implement

    def pprint(self):
        # TODO implement
        pass

    def pplot(self):
        if not self.inter or not self.nointer:
            self.calculate_curves()
        if not self.area:
            self.calculate_area()

        # TODO remove duplicate code, refactor
        cutsinter = [0]
        inter = [0]
        cyclesinter = [(1, 1)]
        for cut in sorted(self.inter.keys()):
            cutsinter.append(cut)
            optim_case = self.inter[cut]
            inter.append(optim_case.results.baseenergycapacity)
            cyclesinter.append(self._get_cycles(optim_case))
        cutsinter.append(1)
        inter.append(self.energycapacity)
        cyclesinter.append((1, 1))

        cutsnointer = [0]
        nointer = [0]
        cyclesnointer = [(1, 1)]
        for cut in sorted(self.nointer.keys()):
            cutsnointer.append(cut)
            optim_case = self.nointer[cut]
            nointer.append(optim_case.results.baseenergycapacity)
            cyclesnointer.append(self._get_cycles(optim_case))
        cutsnointer.append(1)
        nointer.append(self.energycapacity)
        cyclesnointer.append((1, 1))

        ax = plt.figure().add_subplot(1, 1, 1)
        ax.plot([0, self.energycapacity], [0, 1])
        ax.plot(nointer, cutsnointer, linestyle='--')
        ax.plot(inter, cutsinter)

        for x, y, cycles in zip(nointer, cutsnointer, cyclesnointer):
            ax.text(x, y, '{:.2f}, {:.2f}'.format(*cycles),
                    HorizontalAlignment='right', VerticalAlignment='bottom')
        for x, y, cycles in zip(inter, cutsinter, cyclesinter):
            ax.text(x, y, '{:.2f}, {:.2f}'.format(*cycles),
                    HorizontalAlignment='left', VerticalAlignment='top')

        ax.set_ylabel('Cut')
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
