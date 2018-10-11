#!/usr/bin/env python3
"""Functions and methods which process the data of the hybridisation diagram
in a more complex way"""

from scipy.interpolate import griddata
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
from itertools import chain
from operator import itemgetter

from .utility import make_empty_axes


SingularDataInput = namedtuple('SingularDataInput', 'points vals datatype')


def extract_data(hyb, field, copy_peak=False, hook=None, extrahookargs=None):
    points = list()
    vals = list()

    for cut, res in chain(hyb.inter.items(), hyb.nointer.items()):
        chi = cut
        mu = res.basenorm.energy
        point = (chi, mu)
        if point not in points:
            points.append(point)
            vals.append(getattr_deep(res, field))

    for point, res in hyb.area.items():
        if point not in points:
            points.append(point)
            vals.append(getattr_deep(res, field))

    if copy_peak:
        points, vals = copy_highest(points, vals)
    if hook:
        if extrahookargs:
            points, vals = hook(points, vals, *extrahookargs)
        else:
            points, vals = hook(points, vals)

    data = SingularDataInput(points, vals, datatype=field)
    return data


def getattr_deep(start, attr):
    obj = start
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj


def copy_highest(points, vals):
    points = list(points)
    vals = list(vals)
    points_sorted = sorted(points, key=itemgetter(0, 1), reverse=True)
    highest = points_sorted[1]
    highestval = vals[points.index(highest)]
    vals[points.index((1, 1))] = highestval
    return points, vals


class SingularData:
    def __init__(self, points, vals, datatype=None, interp_method='cubic',
                 intgrid=500):
        self.points = list(points)
        self.vals = list(vals)
        self.type = datatype
        self._interpmethod = interp_method
        self._intgrid = intgrid

    @property
    def hybridpotential(self):
        ig = self._intgrid
        xgrid, ygrid = np.meshgrid(np.linspace(1/ig/2, 1-1/ig/2, ig),
                                   np.linspace(1/ig/2, 1-1/ig/2, ig))
        vals = self.interpolant(xgrid.flatten(), ygrid.flatten())
        return 2*(np.count_nonzero(vals)/len(vals))

    @property
    def average(self):
        ig = self._intgrid
        xgrid, ygrid = np.meshgrid(np.linspace(1/ig/2, 1-1/ig/2, ig),
                                   np.linspace(1/ig/2, 1-1/ig/2, ig))
        vals = self.interpolant(xgrid.flatten(), ygrid.flatten())
        return np.sum(vals)/np.count_nonzero(vals)

    def interpolant(self, enorm, cut):
        """Returns callable object taking two argumens enorm, cut"""
        try:
            pointsinter = [(chi, mu) for chi, mu in zip(cut, enorm)]
        except TypeError:
            pointsinter = (cut, enorm)
        # noinspection PyTypeChecker
        return griddata(self.points, self.vals, pointsinter,
                        method=self._interpmethod, fill_value=0)

    def pplot(self, ax=None, intergrid=40, nbins=15, cmap='PuBu', **kwargs):
        y, x = [[*tup] for tup in zip(*self.points)]
        vals = list(self.vals)
        ig = intergrid
        xgrid, ygrid = np.meshgrid(np.linspace(1/ig/2, 1-1/ig/2, ig),
                                   np.linspace(1/ig/2, 1-1/ig/2, ig))
        xgrid = xgrid.flatten()
        ygrid = ygrid.flatten()
        valgrid = self.interpolant(xgrid, ygrid)
        x += list(xgrid[valgrid > 0])
        y += list(ygrid[valgrid > 0])
        vals += list(valgrid[valgrid > 0])

        ax = ax if ax else make_empty_axes()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tricontour(x, y, vals, nbins, colors='k', linewidths=0.5, **kwargs)
        contour = ax.tricontourf(x, y, vals, nbins, cmap=cmap, **kwargs)
        fig = plt.gcf()
        fig.colorbar(contour, ax=ax)


class HybridAnalysis:
    def __init__(self, hyb):
        self.base = StorageResults()
        self.peak = StorageResults()
        self.name = hyb.name

        scaling_factors = self.get_scaling_factors(hyb)
        loss_scale, cycle_scale, charge_scale, psdmax = scaling_factors

        basefields = ['baselosses', 'baseparameters.form',
                      'baseparameters.crest', 'basecycles', 'basepeh',
                      'basepeh', 'basepeh', 'basepeh', 'basepeh', 'basepeh',
                      'basepsd.amplitude', 'basepsd.amplitude',
                      'basepsd.amplidude', 'chargebase']
        peakfields = ['peaklosses', 'peakparameters.form',
                      'peakparameters.crest', 'peakcycles', 'peakpeh',
                      'peakpeh', 'peakpeh', 'peakpeh', 'peakpeh', 'peakpeh',
                      'peakpsd.amplitude', 'peakpsd.amplitude',
                      'peakpsd.amplitude', 'chargepeak']
        attrs = ['losses', 'form', 'crest', 'cycles', 'powerpehlow',
                 'powerpehmed', 'powerpehhigh', 'energypehlow',
                 'energypehmed', 'energypehhigh', 'psdlow', 'psdmed',
                 'psdhigh', 'charge']
        hooks = [None, None, None, None, powerpehlow, powerpehmed,
                 powerpehhigh, energypehlow, energypehmed,
                 energypehhigh, psdlow, psdmed, psdhigh, None]

        hookargs = [loss_scale, None, None, cycle_scale, None, None,
                    psdmax, psdmax, psdmax, charge_scale]

        for attr, field, hook, arg in zip(attrs, basefields, hooks, hookargs):
            data = extract_data(hyb, field, copy_peak=False,
                                hook=hook, extrahookargs=[arg])
            setattr(self.base, attr, SingularDataInput(*data))

        for attr, field, hook, arg in zip(attrs, peakfields, hooks):
            data = extract_data(hyb, field, copy_peak=True,
                                hook=hook, extrahookargs=[arg])
            setattr(self.base, attr, SingularDataInput(*data))

    @staticmethod
    def get_scaling_factors(hyb):
        loss_scale = 1/(hyb.single.losses/hyb.single.dim.energy)
        cycle_scale = 1/hyb.single.cycles
        charge_scale = 1/hyb.single.dim.energy

        # get integral of single psd
        psdint = np.cumsum(hyb.single.psd.amplitude)
        # determine 95% limit
        psd095 = 0.95*psdint[-1]
        psdmaxindex = int(np.flatnonzero(np.array(psdint > psd095))[0] + 1)
        psdval = psdint[-1]

        return loss_scale, cycle_scale, charge_scale, (psdmaxindex, psdval)


class StorageResults:
    def __init__(self):
        self.losses = None
        self.form = None
        self.crest = None
        self.cycles = None
        self.powerpehlow = None
        self.powerpehmed = None
        self.powerpehhigh = None
        self.basepehlow = None
        self.basepehmed = None
        self.basepehhigh = None
        self.psdlow = None
        self.psdmed = None
        self.psdhigh = None
        self.charge = None


# ### Hooks
def psd(points, vals, start, end, whole):
    newvals = [np.cumsum(val[start:end])/whole for val in vals]
    return points, newvals


def psdlow(points, vals, endindexval):
    start, end = 0, int(endindexval[0]/3)
    whole = endindexval[1]
    return psd(points, vals, start, end, whole)


def psdmed(points, vals, endindexval):
    start, end = int(endindexval[0]*1/3 + 1), int(endindexval[1]*2/3)
    whole = endindexval[1]
    return psd(points, vals, start, end, whole)


def psdhigh(points, vals, endindexval):
    start, end = int(endindexval[0]*2/3 + 1), int(len(vals[0]) - 1)
    whole = endindexval[1]
    return psd(points, vals, start, end, whole)


# TODO PEH power assumes symmetric storages (P charge == P discharge)
def powerpehhigh(points, vals):
    newvals = [float(np.sum(val.eweights[0:2]) + np.sum(val.eweights[18:]))
               for val in vals]
    return points, newvals


def powerpehmed(points, vals):
    newvals = [float(np.sum(val.eweights[2:8]) + np.sum(val.eweights[12:18]))
               for val in vals]
    return points, newvals


def powerpehlow(points, vals):
    newvals = [float(np.sum(val.eweights[8:12])) for val in vals]
    return points, newvals


def energypehlow(points, vals):
    newvals = [float(np.sum(val.eweights[0:5])) for val in vals]
    return points, newvals


def energypehmed(points, vals):
    newvals = [float(np.sum(val.eweights[5:15])) for val in vals]
    return points, newvals


def energypehhigh(points, vals):
    newvals = [float(np.sum(val.eweights[15:])) for val in vals]
    return points, newvals
