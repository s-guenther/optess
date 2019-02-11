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
import np


from .optimize_ess import OptimizeSingleESS, OptimizeHybridESS
from .signal import Signal
from .storage import Storage
from .objective import Objective, Solver
from .results import ReducedHybridResults, ReducedSingleResults, \
    single_to_base_results, single_to_peak_results, NoResults


EXACT = Objective('exact', 0)


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
        efeedin = -sigfeedin.integrate()
        esupply = sigsupply.integrate()
        emin = max(0, efeedin-esupply)
        return pmax, emin, efeedin

    def calculate_point(self, storage, objval):
        objective = Objective('energy', objval)
        optsingle = OptimizeSingleESS(self.signal, storage, objective,
                                      self.solver)
        optsingle.solve_pyomo_model()
        if optsingle.results == NoResults:
            energy = None
        else:
            energy = optsingle.results.energycapacity
        redresults = optsingle.results
        return energy, redresults

    def _calculate_point_one_arg(self, storage_objval):
        return self.calculate_point(*storage_objval)

    def find_pareto_front(self, relobjval=0.5, efficiency=0.95,
                          selfdischarge=None, dis_ch_ratio=1, nraster=8,
                          workers=4):
        if selfdischarge is None:
            selfdischarge = 10*self.signal.times[-1]

        # Build storage vector the pareto front shall be solved for
        pmax, emin, efeedin = self.get_estimate_data()
        objval = emin + (efeedin - emin)*relobjval
        powers = np.linspace(0, 2.1*pmax, nraster)
        powers[0] = 1e-6
        storages = [Storage(power, efficiency, selfdischarge)
                    for power in powers]
        # mp.imap only takes one arg, so a two arg function is converted
        # into one arg function (by packing and unpacking)
        stor_objval = [(storage, objval) for storage in storages]
        # First shot to roughly determine pareto front, Solve parallel
        mp.freeze_support()  # Windows Support
        with mp.Pool(processes=workers, initializer=tqdm.set_lock,  # WinSup
                     initargs=(mp.RLock(),)) as pool:
            allopt = list(tqdm(pool.imap(self._calculate_point_one_arg,
                                         stor_objval),
                               total=len(stor_objval)))
        energies = [energy for energy, _ in allopt]
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
        # Upper High Fidelity Raster
        # Lower High Fidelity Raster
        # Final Raster between new pareto front limits
