#!/usr/bin/env python3

import scipy.interpolate as interp
from numpy import trapz, floor
import multiprocessing as mp
from datetime import datetime
import os
import time
from matplotlib import pyplot as plt
import pickle
import textwrap
from tqdm import tqdm
import copy
import numpy as np
from collections import namedtuple


from .optimize_ess import OptimizeSingleESS, OptimizeHybridESS
from .signal import Signal
from .storage import Storage
from .objective import Objective, Solver
from .results import ReducedHybridResults, ReducedSingleResults, \
    single_to_base_results, single_to_peak_results, NoResults


EXACT = Objective('exact', 0)

StoragePara = namedtuple('StoragePara', 'efficiency self_discharge '
                                        'dis_ch_ratio')
SPARA = StoragePara(0.95, None, 1)


class SingleDia:
    # noinspection PyArgumentList
    def __init__(self, signal, pareto_fronts=8, solver='gurobi',
                 name=None, save_opt_results=True):
        self.signal = Signal(signal)
        self.solver = Solver(solver)
        self._save_opt_results = save_opt_results

        if name is None:
            self.name = datetime.now().strftime('SingleDia-%y%m%d-%H%M%S')
        else:
            self.name = str(name)

        if not os.path.isdir(self.name):
            os.mkdir(self.name)

    def get_estimate_data(self):
        pmax = -min(self.signal.vals)
        sigfeedin = self.signal*(self.signal < 0)
        sigsupply = self.signal*(self.signal >= 0)
        efeedin = -sigfeedin.integrate().vals[-1]
        esupply = sigsupply.integrate().vals[-1]
        emin = max(0, efeedin-esupply)
        return pmax, emin, efeedin

    def calculate_point(self, storage, objval):
        objective = Objective('energy', objval)
        optsingle = OptimizeSingleESS(self.signal, storage, objective,
                                      self.solver)
        optsingle.solve_pyomo_model()
        if type(optsingle.results) is NoResults:
            energy = None
        else:
            energy = optsingle.results.energycapacity
        redresults = optsingle.results
        return redresults, energy

    def _calculate_point_one_arg(self, storage_objval):
        return self.calculate_point(*storage_objval)

    def solve_optimizations(self, powers, objval, storagepara=SPARA,
                            workers=4):
        eff = storagepara.efficiency
        pratio = storagepara.dis_ch_ratio
        if storagepara.self_discharge:
            selfdis = storagepara.self_discharge
        else:
            selfdis = self.signal.times[-1]*5
        storages = [Storage([power, power/pratio], eff, selfdis)
                    for power in powers]

        # mp.imap only takes one arg, so a two arg function is converted
        # into one arg function (by packing and unpacking)
        stor_objval = [(storage, objval) for storage in storages]
        # energies = self._parallel_solving(stor_objval, workers=workers)
        energies = self._serial_solving(stor_objval)
        return energies

    def _serial_solving(self, stor_objval):
        energies = list()
        for s_o in tqdm(stor_objval):
            _, energy = self._calculate_point_one_arg(s_o)
            energies.append(energy)
        return energies

    def _parallel_solving(self, stor_objval, workers=4):
        # First shot to roughly determine pareto front, Solve parallel
        mp.freeze_support()  # Windows Support
        with mp.Pool(processes=workers, initializer=tqdm.set_lock,  # WinSup
                     initargs=(mp.RLock(),)) as pool:
            allopt = list(tqdm(pool.imap(self._calculate_point_one_arg,
                                         stor_objval),
                               total=len(stor_objval)))
        energies = [energy for _, energy in allopt]
        return energies

    @staticmethod
    def _filter_invalid(powers, energies):
        # Remove invalid results
        filtpowers, filtenergies = list(), list()
        lastenergy = -1e12  # dummy value
        for power, energy in powers, energies:
            if energy is None:
                continue
            if abs(energy - lastenergy)/energy - 1 < 1e-2:
                break
            filtpowers.append(power)
            filtenergies.append(energy)
            lastenergy = energy
        return filtpowers, filtenergies

    def find_pareto_front(self, relobjval=0.5, storagepara=SPARA, nraster=8,
                          workers=4):
        # Build storage vector the pareto front shall be solved for
        pmax, emin, efeedin = self.get_estimate_data()
        objval = emin + (efeedin - emin)*relobjval

        # Coarse Raster
        coarse_p = np.linspace(0, 2.1*pmax, nraster)
        coarse_p[0] = 1e-6
        coarse_e = self.solve_optimizations(coarse_p, objval, storagepara,
                                            workers)
        coarse_p, coarse_e = self._filter_invalid(coarse_p, coarse_e)

        # Lower High Fidelity Raster
        ptop = coarse_p[0]
        pbot = ptop - (coarse_p[1] - ptop) + 1e-6
        low_p = np.linspace(pbot, ptop, int(np.ceil(nraster/2) + 2))[1:-1]
        # Upper High Fidelity Raster
        ptop = coarse_p[-1]
        pbot = coarse_p[-2]
        high_p = np.linspace(pbot, ptop, int(np.ceil(nraster/2) + 2))[1:-1]
        # Intermediate Raster
        intermed_p = coarse_p[:-1] + np.diff(coarse_p)/2

        # Fine Raster
        fine_p = np.concatenate([low_p, intermed_p, high_p])
        fine_e = self.solve_optimizations(fine_p, objval, storagepara, workers)
        fine_p, fine_e = self._filter_invalid(fine_p, fine_e)

        # Sort all
        powers = [p for _, p in sorted(zip(fine_e, fine_p))]
        energies = sorted(fine_e)
        return energies, powers
