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
import pickle


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
                 name='Hybridisation Diagram', calc_single_at_init=True):
        self.signal = Signal(signal)
        self.storage = Storage(singlestorage)
        self.objective = Objective(objective)
        self.solver = Solver(solver)
        if calc_single_at_init:
            self.single = self.calculate_single()
            self.energycapacity = self.single.results.energycapacity
        else:
            self.single = None
            self.energycapacity = None
        # self.inter = OnTheFlyDict(self, 'inter')
        # self.nointer = OnTheFlyDict(self, 'nointer')
        self.inter = dict()
        self.nointer = dict()
        self.area = dict()
        self.name = str(name)

        self.powercapacity = self.storage.power.max

    def calculate_single(self):
        single = OptimizeSingleESS(self.signal, self.storage,
                                   self.objective, self.solver)
        single.solve_pyomo_model(clear_model=True)
        self.energycapacity = single.results.energycapacity
        return single

    def calculate_cut(self, cut, strategy='inter', add_to_internal_list=True):
        signal = self.signal
        base = cut*self.storage
        peak = (1 - cut)*self.storage
        objective = self.objective
        solver = self.solver

        optim_case = OptimizeHybridESS(signal, base, peak, objective,
                                       strategy, solver)
        optim_case.solve_pyomo_model(clear_model=True)

        if add_to_internal_list:
            if strategy == 'inter':
                self.inter[cut] = optim_case
            elif strategy == 'nointer':
                self.nointer[cut] = optim_case
        return optim_case

    def _parallel_both_cuts(self, cut):
        print('Starting cut={}'.format(cut))
        inter = self.calculate_cut(cut, 'inter',
                                   add_to_internal_list=False)
        nointer = self.calculate_cut(cut, 'nointer',
                                     add_to_internal_list=False)
        print('Ending cut={}'.format(cut))
        return cut, inter, nointer

    def calculate_curves(self, cuts=(0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 0.99)):
        with mp.Pool() as pool:
            res = pool.map(self._parallel_both_cuts, cuts)
        print('Finished parallel Hybrid Curve Calculation')

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

        # Filter point list
        points = itertools.product(energies, powers)
        filteredpoints = []
        for point in points:
            if self.is_point_in_area(*point):
                filteredpoints.append(point)
        filteredpoints = tuple(filteredpoints)

        res = []
        for point in filteredpoints:
            res.append(self._parallel_point(point))
        print('Finished Parallel Area Calculation')

        for energy, power, optim_case in res:
            self.area[(energy, power)] = optim_case

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

    def _parallel_point(self, point):
        print('Start calculating area point {}'.format(point))
        optim_case = self.calculate_point(*point, add_to_internal_list=False)
        print('End calculating area point {}'.format(point))
        energy, power = point[0], point[1]
        return energy, power, optim_case

    def calculate_point(self, energy, power, add_to_internal_list=True):
        """The optimisation problem is solved for this point defined by
        power and energy."""
        cut = power/self.powercapacity
        base = cut*self.storage
        peak = (1-cut)*self.storage
        optim_case = OptimizeHybridESS(self.signal, base, peak,
                                       self.objective, solver=self.solver)

        baseenergy = energy
        peakenergy = self.energycapacity - energy
        optim_case.solve_pyomo_model(baseenergy, peakenergy, clear_model=True)

        if add_to_internal_list:
            self.area[(power, energy)] = optim_case

        return optim_case

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

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([0, self.energycapacity], [0, self.powercapacity],
                color='black')
        ax.plot(nointer, powernointer, linestyle='--', color='orange',
                linewidth=2)
        ax.plot(inter, powerinter, color='black')

        xvec, yvec, cycvec = list(), list(), list()
        for x, y, cycles in zip(nointer, powernointer, cyclesnointer):
            ax.text(x, y, '{:.2f}'.format(cycles[1]))
            xvec.append(x)
            yvec.append(y)
            cycvec.append(cycles[0])
        for x, y, cycles in zip(inter, powerinter, cyclesinter):
            ax.text(x, y, '{:.2f}'.format(cycles[1]))
            xvec.append(x)
            yvec.append(y)
            cycvec.append(cycles[0])
        for (x, y), opt_case in self.area.items():
            cbase, cpeak = self._get_cycles(opt_case)
            ax.text(x, y, '{:.2f}'.format(cpeak))
            xvec.append(x)
            yvec.append(y)
            cycvec.append(cbase)
        cycsingle = self.single.results.power.cycles(self.energycapacity)
        xvec.append(self.energycapacity)
        yvec.append(self.powercapacity)
        cycvec.append(cycsingle)
        xvec.append(0)
        yvec.append(0)
        cycvec.append(self.storage.efficiency.charge)
        ax.tricontour(xvec, yvec, cycvec, 14, colors='k', linewidths=0.5)
        contour = ax.tricontourf(xvec, yvec, cycvec, 14, cmap='PuBu')
        fig.colorbar(contour, ax=ax)

        ax2 = fig.add_axes((0.16, 0.76, 0.24, 0.10))
        ax3 = fig.add_axes((0.16, 0.64, 0.24, 0.10))
        ax4 = fig.add_axes((0.16, 0.52, 0.24, 0.10))
        self.single.pplot(ax=(ax2, ax3, ax4))

        ax.set_ylabel('Power')
        ax.set_xlabel('Energy')
        ax.autoscale(tight=True)

    def save(self, filename):
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'hyb'

        savedict = dict()
        names = ['signal', 'storage', 'objective', 'solver', 'single', 'name',
                 'inter', 'nointer', 'area', 'powercapacity', 'energycapacity']
        for name in names:
            savedict[name] = getattr(self, name)

        with open(sep.join([filename, fileend]), 'wb') as file:
            pickle.dump(savedict, file)

    @classmethod
    def load(cls, filename):
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'hyb'

        with open(sep.join([filename, fileend]), 'rb') as file:
            savedict = pickle.load(file)

        opt_case = HybridDia(savedict['signal'], savedict['storage'],
                             savedict['objective'], savedict['solver'],
                             savedict['name'], calc_single_at_init=False)

        remaining = ['single', 'energycapacity', 'powercapacity', 'inter',
                     'nointer', 'area']
        for name in remaining:
            setattr(opt_case, name, savedict[name])

        return opt_case

    @staticmethod
    def _get_cycles(optim_case):
        """Returns base and peak cycles as tuple"""
        energybase = optim_case.results.baseenergycapacity
        energypeak = optim_case.results.peakenergycapacity
        basepower = optim_case.results.base + optim_case.results.baseinter
        peakpower = optim_case.results.peak + optim_case.results.peakinter
        return basepower.cycles(energybase), peakpower.cycles(energypeak)

# TODO __repr__
