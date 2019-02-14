#!/usr/bin/env python3

import multiprocessing as mp
from datetime import datetime
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import namedtuple, OrderedDict
import time
import pickle


from .optimize_ess import OptimizeSingleESS
from .signal import Signal
from .storage import Storage
from .objective import Objective, Solver
from .results import NoResults


EXACT = Objective('exact', 0)

StoragePara = namedtuple('StoragePara', 'efficiency self_discharge '
                                        'dis_ch_ratio')
SPARA = StoragePara(0.95, None, 1)


class SingleDia:
    # noinspection PyArgumentList
    def __init__(self, signal, rel_obj_fronts=(0.1, 0.3, 0.5, 0.7, 0.9),
                 resolution=8, storagepara=SPARA, solver='gurobi', name=None,
                 save_opt_results=True):
        self.signal = Signal(signal)
        self.solver = Solver(solver)
        self._save_opt_results = save_opt_results
        self.rel_obj_fronts = rel_obj_fronts
        self.resolution = resolution
        self.storagepara = storagepara
        self.results = OrderedDict()

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
        emin = max(0, esupply - efeedin)
        return pmax, emin, esupply

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
        storages = [Storage([-power, power/pratio], eff, selfdis)
                    for power in powers]

        # mp.imap only takes one arg, so a two arg function is converted
        # into one arg function (by packing and unpacking)
        stor_objval = [(storage, objval) for storage in storages]
        # energies = self._parallel_solving(stor_objval, workers=workers)
        energies = self._parallel_solving(stor_objval)
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
        for power, energy in zip(powers, energies):
            if energy is None:
                continue
            if abs(energy - lastenergy)/energy < 1e-2:
                break
            filtpowers.append(power)
            filtenergies.append(energy)
            lastenergy = energy
        return filtpowers, filtenergies

    def find_pareto_front(self, relobjval=0.5, storagepara=None, nraster=None,
                          workers=4):
        if storagepara is None:
            storagepara = self.storagepara
        if nraster is None:
            nraster = self.resolution

        # Build storage vector the pareto front shall be solved for
        pmax, emin, esupply = self.get_estimate_data()
        objval = emin + (esupply - emin)*(1 - relobjval)

        msg = 'Calculate Pareto Front for \n' \
              '    relative objective = {}\n' \
              '    absolute objective = {}\n' \
              '    min objective      = {}\n' \
              '    max objective      = {}\n' \
              '----------------------------------------------------------\n'
        msgvals = (relobjval, objval, emin, esupply)
        print(msg.format(*msgvals), flush=True)
        time.sleep(0.01)

        # Coarse Raster
        print('Calculate Coarse Raster...', flush=True)
        time.sleep(0.01)
        coarse_p = np.linspace(0.1, 2.0*pmax, nraster)
        coarse_e = self.solve_optimizations(coarse_p, objval, storagepara,
                                            workers)
        coarse_p, coarse_e = self._filter_invalid(coarse_p, coarse_e)

        if len(coarse_p) > 1:
            # Lower High Fidelity Raster
            ptop = coarse_p[0]
            pbot = ptop - (coarse_p[1] - ptop) + 1e-6
            pbot = pbot if pbot > 0 else 1e-6
            low_p = np.linspace(pbot, ptop, int(np.ceil(nraster/2) + 2))[1:-1]
            # Upper High Fidelity Raster
            ptop = coarse_p[-1]
            pbot = coarse_p[-2]
            high_p = np.linspace(pbot, ptop, int(np.ceil(nraster/2) + 2))[1:-1]
            # Intermediate Raster
            intermed_p = coarse_p[:-1] + np.diff(coarse_p)/2
            fine_p = np.concatenate([low_p, intermed_p, high_p])
        elif len(coarse_p) == 1:
            nn = int(np.ceil(nraster*1.5))
            fine_p = np.linspace(1e-2*coarse_p[0], nn/(nn+1)*coarse_p[0], nn)
        else:
            return [], []

        # Fine Raster
        time.sleep(0.01)
        print('Calculate Fine Raster...', flush=True)
        time.sleep(0.01)
        fine_e = self.solve_optimizations(fine_p, objval, storagepara, workers)
        fine_p, fine_e = self._filter_invalid(fine_p, fine_e)

        # Concat fine and coarse
        powers = np.concatenate([fine_p, coarse_p])
        energies = np.concatenate([fine_e, coarse_e])

        # Sort all
        ind = np.flip(np.argsort(energies))
        powers = powers[ind]
        energies = energies[ind]
        powers, energies = self._filter_invalid(powers, energies)
        time.sleep(0.01)
        print('\n----------------------------------------------------------')
        print('... done\n')
        return energies, powers

    def compute(self, *args, **kwargs):
        return self.compute_parallel(*args, **kwargs)

    def compute_parallel(self, workers=4):
        for relobj in self.rel_obj_fronts:
            args = (relobj, self.storagepara, self.resolution, workers)
            res = self.find_pareto_front(*args)
            self.results[relobj] = res
        print('----------------\n... all done\n', flush=True)

    def pplot(self, ax=None):
        if not self.results:
            print('Compute results with singledia.compute() first')
            return
        if ax is None:
            ax = plt.figure().add_subplot(1, 1, 1)
            ax.set_xlabel('Energy')
            ax.set_ylabel('Power')

        for relobj, (energies, powers) in self.results.items():
            ax.plot(energies, powers, color='b')

    def save(self, filename=None):
        if filename is None:
            filename = self.name
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'sgl'

        with open(sep.join([filename, fileend]), 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        sep = '.'
        try:
            filename, fileend = filename.split(sep)
        except ValueError:
            filename, fileend = filename, 'sgl'

        with open(sep.join([filename, fileend]), 'rb') as file:
            sdia = pickle.load(file)
        return sdia

