#!/usr/bin/env python3

import itertools
import scipy.interpolate as interp
from numpy import linspace, trapz
import multiprocessing as mp
from datetime import datetime
import os
import time
from matplotlib import pyplot as plt
import pickle
import textwrap
from tqdm import tqdm
import copy

from .optimize_ess import OptimizeSingleESS, OptimizeHybridESS
from .signal import Signal
from .storage import Storage
from .objective import Objective, Solver
from .results import ReducedHybridResults, ReducedSingleResults, \
                     single_to_base_results, single_to_peak_results


DIMTOL = 1 + 1e-6
CUTS = (0.10, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95)
CURVES = 4


def print_resources():
    ns = [1000, 2000, 4000, 6000, 9000, 13000, 20000, 30000, 40000, 55000,
          80000, 115000, 140000, 175000, 200000, 230000]
    mems = [0.3, 0.5, 0.8, 1.2, 1.8, 3.2, 4.3, 7.3, 10.6, 14.8, 21, 30.2, 36,
            45.5, 51.5, 61]
    cpus = [1, 1, 1, 1, 1, 2, 2, 3, 4, 8, 8, 12, 12, 16, 16, 16]
    intromsg = "Depending on the length 'n' of the dataset or load profile, " \
               "a different amount of memory/RAM 'mem' is allocated by the " \
               "optimisation and a different amount of CPUs 'cpu' is " \
               "advised to use. Performing the optimisation with less mem " \
               "will result in swapping and drastically increased " \
               "calculation times. Performing the optimisation with less " \
               "cpu is not as severe but also noticable. Be aware that the " \
               "operating system also uses some RAM. If you compute with " \
               "parallel workers, be aware that each worker needs the " \
               "specified amount of cpu and ram. If you have a cluster with " \
               "a torque scheduling system available ensure that the " \
               "correct '.torquesetup' file is provided."
    for line in textwrap.wrap(intromsg):
        print(line)
    print('\n')
    print('{:>8}{:>8}{:>7}'.format('n', 'mem', 'cpu'))
    print('-'*23)
    for n, mem, cpu in zip(ns, mems, cpus):
        print('{:>8}{:>8}{:>7}'.format(n, mem, cpu))


class PointOutsideAreaError(ValueError):
    """Raised if a point within the Hybridisation Area shall be calculated,
    but the point is outside"""
    pass


class HybridDia:
    # noinspection PyArgumentList
    def __init__(self, signal, singlestorage, objective, solver='gurobi',
                 name=None, save_opt_results=True):
        self.signal = Signal(signal)
        self.storage = Storage(singlestorage)
        self.objective = Objective(objective)
        self.solver = Solver(solver)
        self._save_opt_results = save_opt_results

        if name is None:
            self.name = datetime.now().strftime('HybDia-%y%m%d-%H%M%S')
        else:
            self.name = str(name)

        if not os.path.isdir(self.name):
            os.mkdir(self.name)

        self.energycapacity = None
        self.powercapacity = self.storage.power.max

        self.single = None
        self.inter = dict()
        self.nointer = dict()
        self.area = dict()

    # --- Properties ---
    @property
    def hybridpotential(self):
        cuts = sorted(self.inter.keys())
        energies = []
        for cut in cuts:
            energies.append(self.inter[cut].basedim.energy)
        singleenergy = self.energycapacity
        return 2/singleenergy*trapz(energies, x=cuts) - 1

    @property
    def reloadpotential(self):
        cuts = sorted(self.nointer.keys())
        energies = []
        for cut in cuts:
            energies.append(self.nointer[cut].basedim.energy)
        singleenergy = self.energycapacity
        rpot = 2/singleenergy*trapz(energies, x=cuts) - 1
        return self.hybridpotential - rpot

    # --- Compute Function to calculate complete diagram ---
    def compute(self, cuts=CUTS, curves=CURVES):
        """This function computes: (1) The single optimizsation to gain the
        upper right point of the hybridisation diagram; (2) The
        hybridisation curve at specified cuts; (3) Points within the area to
        determine cyclisation.
        For standard values of cuts, see module.CUTS.
        For generating the hybridisation diagram, many separate and mostly
        independend optimisations have to be solved. For this process,
        3 different computation routines are available: 'self.compute_serial',
        'self.compute_parallel' or 'self.compute_torque'. Depending on the
        system you are working on and dataset you are working with
        computation routines will differ in speed. This function deferres
        execution to 'self.compute_serial'. See help of specialized
        functions for more information."""
        self.compute_serial(cuts=cuts, curves=curves)

    def compute_serial(self, cuts=CUTS, curves=CURVES):
        """See 'self.compute' for more information. Each optimisation is
        done one after another. This will be the slowest procedure if enough
        hardware ressources are available. See 'module.print_resources()' to
        estimate required resources."""
        # 10ms sleep to ensure correct text flushing of print and tqdm
        print('Calculate Single...', flush=True)
        time.sleep(0.01)
        for _ in tqdm([None]):
            self.calculate_single()
        self._add_extreme_points()

        time.sleep(0.01)
        print('Calculate hybrid curve with inter-storage power flow...',
              flush=True)
        time.sleep(0.01)
        for cut in tqdm(cuts):
            self.calculate_point_at_curve(cut, 'inter')

        time.sleep(0.01)
        print('Calculate hybrid curve without inter-storage power flow...',
              flush=True)
        time.sleep(0.01)
        for cut in tqdm(cuts):
            self.calculate_point_at_curve(cut, 'nointer')
        points = self.get_points_in_area(curves=curves)

        time.sleep(0.01)
        print('Calculate points within area...', flush=True)
        time.sleep(0.01)
        for point in tqdm(points):
            self.calculate_point_in_area(*point)

        time.sleep(0.01)
        print('... all done', flush=True)

    def compute_parallel(self, cuts=CUTS, curves=CURVES, workers=4):
        """See 'self.compute' for more information. Optimisations are
        computed in parallel with a pool of 'workers'. See
        'module.print_ressources()' to determine the right amount of
        workers."""
        # 10ms sleep to ensure correct text flushing of print and tqdm

        # --- Calculate Single ---
        print('Calculate Single...', flush=True)
        time.sleep(0.01)
        for _ in tqdm([None]):
            self.calculate_single()
        self._add_extreme_points()

        # --- Calculate Hybrid Curve w/ inter-storage power flow ---
        time.sleep(0.01)
        print('Calculate hybrid curve with inter-storage power flow...',
              flush=True)
        time.sleep(0.01)

        mp.freeze_support()  # for Windows support

        with mp.Pool(processes=workers, initializer=tqdm.set_lock,  # WinSup
                     initargs=(mp.RLock(),)) as pool:
            allinter = list(tqdm(pool.imap(self._calculate_inter, cuts),
                                 total=len(cuts)))
        for cut, inter in allinter:
            self.inter[cut] = inter

        # --- Calculate Hybrid Curve w/o inter-storage power flow ---
        time.sleep(0.01)
        print('Calculate hybrid curve without inter-storage power flow...',
              flush=True)
        time.sleep(0.01)

        with mp.Pool(processes=workers, initializer=tqdm.set_lock,  # WinSup
                     initargs=(mp.RLock(),)) as pool:
            allnointer = list(tqdm(pool.imap(self._calculate_nointer, cuts),
                                   total=len(cuts)))
        for cut, nointer in allnointer:
            self.nointer[cut] = nointer

        points = self.get_points_in_area(curves=curves)

        # --- Calculates points in area ---
        time.sleep(0.01)
        print('Calculate points within area...', flush=True)
        time.sleep(0.01)

        with mp.Pool(processes=workers, initializer=tqdm.set_lock,  # WinSup
                     initargs=(mp.RLock(),)) as pool:
            allarea = list(tqdm(pool.imap(self._calculate_area_point, points),
                                total=len(points)))
        for point, area in allarea:
            self.area[point] = area

        # --- Finish line ---
        time.sleep(0.01)
        print('... all done', flush=True)

    def compute_torque(self, cuts=CUTS, curves=CURVES,
                       setupfile=None, wt=1.0, mem=1.0):
        """See 'self.compute' for more information. The single optimisations
        are delegated to a torque batch system at a cluster, assuming that the
        shell command 'qsub' is available. Provide a path to 'setupfile',
        else, the file will be searched in the current working directory.
        'wt' scales the predefined walltime, 'mem' the predefined RAM."""
        pass

    # --- unary calculation function, calculates only a single point ---
    def calculate_single(self, add_to_dia=True):
        single = OptimizeSingleESS(self.signal, self.storage,
                                   self.objective, self.solver)
        single.solve_pyomo_model()
        self.energycapacity = single.results.energycapacity
        results = ReducedSingleResults(single, savepath=self.name,
                                       save_to_disc=self._save_opt_results)
        if add_to_dia:
            self.single = results
        return results

    def calculate_point_at_curve(self, cut, strategy='inter', add_to_dia=True):
        signal = self.signal
        base = cut*self.storage*DIMTOL
        peak = (1-cut)*self.storage*DIMTOL
        objective = self.objective
        solver = self.solver

        optim_case = OptimizeHybridESS(signal, base, peak, objective,
                                       strategy, solver)
        optim_case.solve_pyomo_model()

        results = ReducedHybridResults(optim_case, savepath=self.name,
                                       save_to_disc=self._save_opt_results)

        if add_to_dia:
            if strategy == 'inter':
                self.inter[cut] = results
            elif strategy == 'nointer':
                self.nointer[cut] = results
        return results

    def calculate_point_in_area(self, cut, enorm, add_to_dia=True):
        """The optimisation problem is solved for this point defined by
        power and energy."""
        base = cut * self.storage * DIMTOL
        peak = (1 - cut) * self.storage * DIMTOL
        optim_case = OptimizeHybridESS(self.signal, base, peak,
                                       self.objective, solver=self.solver)

        baseenergy = enorm * self.energycapacity * DIMTOL
        peakenergy = (1 - enorm) * self.energycapacity * DIMTOL
        optim_case.solve_pyomo_model(baseenergy, peakenergy)

        results = ReducedHybridResults(optim_case, savepath=self.name,
                                       save_to_disc=self._save_opt_results)

        if add_to_dia:
            self.area[(cut, enorm)] = results

        return results

    # --- auxilliary functions ---
    def get_points_in_area(self, curves=CURVES):
        """Returns points which are within the area enclose by
        the hybridisation curve and the single storage specific power line.
        Points are returned between the power cut points and the specific
        power line (which is also included) in an equidistant manner for
        this power cut. Parameter 'curves' describes the number of points
        between the hybridisation curve and the specific power line per cut"""
        cuts = set(self.inter.keys())
        cuts.remove(0)
        cuts.remove(1)
        points = set()
        for cut in cuts:
            single = cut
            base = self.inter[cut].basenorm.energy
            for ii in range(curves):
                enorm = single + (base - single)/curves*ii
                points.add((cut, enorm))
        return list(points)

    def is_point_in_area(self, cut, enorm):
        """Return True if a point is within the hybridisation area."""
        if cut < 0 or cut > 1 or enorm < 0 or enorm > 1:
            return False

        # Interpolate Hybridisation Curve with given points
        cuts = sorted(self.inter.keys())
        enorms = [self.inter[hcut].basenorm.energy for hcut in cuts]
        hcurve = interp.interp1d(cuts, enorms, 'linear')

        minenorm = cut  # left side of area at spec. cut
        maxenorm = hcurve(cut)  # right side of area at specified cut

        return minenorm <= enorm <= maxenorm

    def _calculate_inter(self, cut):
        return cut, self.calculate_point_at_curve(cut, 'inter', False)

    def _calculate_nointer(self, cut):
        return cut, self.calculate_point_at_curve(cut, 'nointer', False)

    def _calculate_area_point(self, point):
        return point, self.calculate_point_in_area(*point, False)

    def _add_extreme_points(self):
        """Adds to inter and nointer dictionary the power cuts 0 and 1 which
        can be derived from single point calculation without the need to
        calculate it from hybrid optimisation"""
        # Use duck typing and generate a fake reduced hybrid result with
        # the help of reduced single results
        self.inter[0] = single_to_peak_results(self.single)
        self.inter[1] = single_to_base_results(self.single)
        self.nointer[0] = copy.copy(self.inter[0])
        self.nointer[1] = copy.copy(self.inter[1])

    # --- output/save/load functions ---
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
            results = self.inter[cut]
            inter.append(results.basedim.energy)
            cyclesinter.append((results.basecycles, results.peakcycles))

        powernointer = []
        nointer = []
        cyclesnointer = []
        for cut in sorted(self.nointer.keys()):
            powernointer.append(cut*self.powercapacity)
            results = self.nointer[cut]
            nointer.append(results.basedim.energy)
            cyclesnointer.append((results.basecycles, results.peakcycles))

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
        for (y, x), results in self.area.items():
            cbase, cpeak = (results.basecycles, results.peakcycles)
            ax.text(x*self.energycapacity, y*self.powercapacity,
                    '{:.2f}'.format(cpeak))
            xvec.append(x*self.energycapacity)
            yvec.append(y*self.powercapacity)
            cycvec.append(cbase)
        cycsingle = self.single.cycles
        xvec.append(self.energycapacity)
        yvec.append(self.powercapacity)
        cycvec.append(cycsingle)
        xvec.append(0)
        yvec.append(0)
        cycvec.append(self.storage.efficiency.charge)
        ax.tricontour(xvec, yvec, cycvec, 14, colors='k', linewidths=0.5)
        contour = ax.tricontourf(xvec, yvec, cycvec, 14, cmap='PuBu')
        fig.colorbar(contour, ax=ax)

        try:
            single = self.single.load_all_results()
        except FileNotFoundError:
            pass
        else:
            ax2 = fig.add_axes((0.16, 0.76, 0.24, 0.10))
            ax3 = fig.add_axes((0.16, 0.64, 0.24, 0.10))
            ax4 = fig.add_axes((0.16, 0.52, 0.24, 0.10))
            single.pplot(ax=(ax2, ax3, ax4))

        ax.set_ylabel('Power')
        ax.set_xlabel('Energy')
        ax.autoscale(tight=True)

    def save(self, filename=None):
        if filename is None:
            filename = self.name
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'hyb'

        with open(sep.join([filename, fileend]), 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'hyb'

        with open(sep.join([filename, fileend]), 'rb') as file:
            opt_case = pickle.load(file)
        return opt_case

# TODO __repr__
