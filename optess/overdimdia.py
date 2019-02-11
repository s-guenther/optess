#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from matplotlib import pyplot as plt

from .signal import Signal
from .objective import Objective, Solver, Strategy
from .storage import FullStorage
from .optimize_ess import OptimizeHybridESS
from .results import ReducedHybridResults


class UnknownPlotTypeError(ValueError):
    pass


class UnknownComputeTypeError(ValueError):
    pass


class NoResultsComputedError(LookupError):
    pass


# --- Superclass

class AbstractOverdimDia(ABC):
    # noinspection PyArgumentList
    def __init__(self, signal, singlestor, reduced_hybrid_result, objective,
                 strategy='inter', solver='gurobi', name=None,
                 save_opt_results=True):
        self.signal = Signal(signal)
        self.single = singlestor
        self.objective = Objective(objective)
        self.solver = Solver(solver)
        self.strategy = Strategy(strategy)
        self.name = name
        self.orig_res = reduced_hybrid_result
        self._save_opt_results = save_opt_results

        # Build Full Storages
        red = reduced_hybrid_result
        self.base = FullStorage(red.basedim.energy,
                                red.basedim.power,
                                singlestor.efficiency,
                                singlestor.selfdischarge)
        self.peak = FullStorage(red.peakdim.energy,
                                red.peakdim.power,
                                singlestor.efficiency,
                                singlestor.selfdischarge)

        # Result Fields
        self.meshx = None
        self.meshy = None
        self.meshstor = None
        self.meshres = None
        self.meshcycle = None

    def get_init_args(self):
        initargs = (self.signal, self.single, self.orig_res, self.objective,
                    self.strategy, self.solver, None)
        return initargs

    def build_mesh_arrays(self, overdimfactor=(1.5, 1.5), nraster=(4, 4),
                          start=(1, 1)):
        try:
            overdimfactor[0]
        except TypeError:
            overdimfactor = [overdimfactor]*2
        try:
            nraster[0]
        except TypeError:
            nraster = [nraster]*2
        try:
            start[0]
        except TypeError:
            start = [start]*2

        x = np.linspace(start[0], overdimfactor[0], nraster[0])
        y = np.linspace(start[1], overdimfactor[1], nraster[1])
        self.meshx, self.meshy = np.meshgrid(x, y)
        self.meshstor = np.empty(self.meshx.shape, dtype=object)
        self.meshres = np.empty(self.meshx.shape, dtype=object)
        self.meshcycle = np.empty(self.meshx.shape)

        for irow in range(self.meshx.shape[0]):
            for icol in range(self.meshx.shape[1]):
                # Imitates GoF Template Pattern --> delegated to subclass
                overxy = self.meshx[irow, icol], self.meshy[irow, icol]
                self.meshstor[irow, icol] = self.build_storage_pair(*overxy)

    @abstractmethod
    def build_storage_pair(self, overx, overy):
        # GoF Template function delegated to subclasses
        pass

    def compute(self, *args, **kwargs):
        self.compute_parallel(*args, **kwargs)

    def compute_serial(self):
        self._check_mesh_arrays()

        fullstorageslist = self.meshstor.flatten()
        for flatind, storagepair in enumerate(tqdm(fullstorageslist)):
            res, cycles = self.calculate_single_point(storagepair)
            ind = np.unravel_index(flatind, self.meshstor.shape)
            self.meshres[ind] = res
            self.meshcycle[ind] = cycles

    def compute_parallel(self, workers=4):
        self._check_mesh_arrays()

        mp.freeze_support()  # for Windows Support
        with mp.Pool(processes=workers, initializer=tqdm.set_lock,  # WinSup
                     initargs=(mp.RLock(),)) as pool:
            fullstorages = self.meshstor.flatten()
            res_cycles = list(tqdm(pool.imap(self.calculate_single_point,
                                             fullstorages),
                                   total=len(fullstorages)))
        for flatind, (res, cycles) in enumerate(res_cycles):
            ind2d = np.unravel_index(flatind, self.meshstor.shape)
            self.meshres[ind2d] = res
            self.meshcycle[ind2d] = cycles

    def _check_mesh_arrays(self):
        anyempty = (self.meshx is None or
                    self.meshy is None or
                    self.meshstor is None)
        if anyempty:
            self.build_mesh_arrays()

    def calculate_single_point(self, fullstorages):
        fullbase, fullpeak = fullstorages
        opt = OptimizeHybridESS(self.signal, fullbase, fullpeak,
                                self.objective, self.strategy, self.solver)
        opt.solve_pyomo_model(fullbase.energy, fullpeak.energy)
        res = ReducedHybridResults(opt, savepath=self.name,
                                   save_to_disc=self._save_opt_results)
        cycles = res.basecycles
        return res, cycles/self.orig_res.basecycles

    def pplot(self, ax=None):
        if ax is None:
            ax = plt.figure().add_subplot(1, 1, 1)
        cont = ax.contour(self.meshx, self.meshy, self.meshcycle)
        plt.clabel(cont, inline=True, fontsize=8)
        self._label_axes(ax)
        ax.set_title(self.name)

    @abstractmethod
    def _label_axes(self, ax):
        pass


# --- Subclasses

class OverdimDiaBase(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Base Energy/Power @ Peak=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        baseover = FullStorage(bb.energy*overx,
                               [bb.power.min*overy, bb.power.max*overy],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = self.peak
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Base Energy @ Peak=const.')
        ax.set_ylabel('Base Power @ Peak=const.')


class OverdimDiaPeak(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Peak Energy/Power @ Base=const.'

    def build_storage_pair(self, overx, overy):
        baseover = self.base
        pp = self.peak
        peakover = FullStorage(pp.energy*overx,
                               [pp.power.min*overy, pp.power.max*overy],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Peak Energy @ Base=const.')
        ax.set_ylabel('Peak Power @ Base=const.')


class OverdimDiaDim(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Base, Peak @ E/P=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        pp = self.peak
        baseover = FullStorage(bb.energy*overx,
                               [bb.power.min*overx, bb.power.max*overx],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = FullStorage(pp.energy*overy,
                               [pp.power.min*overy, pp.power.max*overy],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Base Energy @ P/E=const.')
        ax.set_ylabel('Peak Energy @ P/E=const.')


class OverdimDiaEnergy(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Base, Peak Energy @ P=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        pp = self.peak
        baseover = FullStorage(bb.energy*overx,
                               [bb.power.min, bb.power.max],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = FullStorage(pp.energy*overy,
                               [pp.power.min, pp.power.max],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Base Energy @ Pb, Pp=const.')
        ax.set_ylabel('Peak Energy @ Pb, Pp=const.')


class OverdimDiaPower(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Base, Peak Power @ E=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        pp = self.peak
        baseover = FullStorage(bb.energy,
                               [bb.power.min*overx, bb.power.max*overx],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = FullStorage(pp.energy*overy,
                               [pp.power.min*overy, pp.power.max*overy],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Base Power @ Eb, Ep=const.')
        ax.set_ylabel('Peak Power @ Eb, Ep=const.')


class OverdimDiaCrossEbPp(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Cross EBase, PPeak @ Pb,Ep=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        pp = self.peak
        baseover = FullStorage(bb.energy*overx,
                               [bb.power.min, bb.power.max],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = FullStorage(pp.energy,
                               [pp.power.min*overy, pp.power.max*overy],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Base Energy @ Ep, Pb=const.')
        ax.set_ylabel('Peak Power @ Ep, Pb=const.')


class OverdimDiaCrossEpPb(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Cross EPeak, PBase @ Pp,Eb=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        pp = self.peak
        baseover = FullStorage(bb.energy,
                               [bb.power.min*overy, bb.power.max*overy],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = FullStorage(pp.energy*overx,
                               [pp.power.min, pp.power.max],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Peak Energy @ Eb, Pp=const.')
        ax.set_ylabel('Base Power @ Eb, Pp=const.')


class OverdimDiaMixed(AbstractOverdimDia):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.name is None:
            self.name = 'Overdimensioning Mixed Eb, Ep @ Pb, Pp/Ep=const.'

    def build_storage_pair(self, overx, overy):
        bb = self.base
        pp = self.peak
        baseover = FullStorage(bb.energy*overx,
                               [bb.power.min, bb.power.max],
                               bb.efficiency,
                               bb.selfdischarge)
        peakover = FullStorage(pp.energy*overy,
                               [pp.power.min*overy, pp.power.max*overy],
                               pp.efficiency,
                               pp.selfdischarge)
        return baseover, peakover

    def _label_axes(self, ax):
        ax.set_xlabel('Base Energy @ Pb, Pp/Ep=const.')
        ax.set_ylabel('Peak Energy @ Pb, Pp/Ep=const.')


OverdimDia = OverdimDiaMixed
