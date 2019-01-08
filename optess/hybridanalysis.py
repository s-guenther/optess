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


def extract_data(hyb, field, copy_peak=False, hook=None, extrahookarg=None):
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
        if extrahookarg:
            points, vals = hook(points, vals, extrahookarg)
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
    def __init__(self, points, vals, datatype=None, interp_method='linear',
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
        nonzerovals = np.count_nonzero(vals)
        nonzerovals = nonzerovals if nonzerovals else 1
        return np.sum(vals)/nonzerovals

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
        # only double values saved in self.single, no SingularData object
        self.single = StorageResults()
        self.name = hyb.name

        self._populate_single(hyb)
        self._populate_base_and_peak(hyb)

    def _populate_single(self, hyb):
        scaling_factors = self._get_scaling_factors(hyb)
        loss_scale, cycle_scale, charge_scale, form_scale, crest_scale, \
            ppeh_scale, epeh_scale, psd_scale = scaling_factors

        self.single.losses = hyb.single.losses/hyb.single.dim.energy
        self.single.form = hyb.single.signalparameters.form
        self.single.crest = hyb.single.signalparameters.crest
        self.single.cycles = hyb.single.cycles
        self.single.powerpehlow = ppeh_scale[0]
        self.single.powerpehmed = ppeh_scale[1]
        self.single.powerpehhigh = ppeh_scale[2]
        self.single.energypehlow = epeh_scale[0]
        self.single.energypehmed = epeh_scale[1]
        self.single.energypehhigh = epeh_scale[2]
        self.single.psdlow = psd_scale[0][1]
        self.single.psdmed = psd_scale[1][1]
        self.single.psdhigh = psd_scale[2][1]
        self.single.charge = 0

    def _populate_base_and_peak(self, hyb):
        scaling_factors = self._get_scaling_factors(hyb)
        loss_scale, cycle_scale, charge_scale, form_scale, crest_scale, \
            ppeh_scale, epeh_scale, psd_scale = scaling_factors

        basefields = ['baselosses', 'basecycles', 'chargebase',
                      'baseparameters.form', 'baseparameters.crest',
                      'basepeh', 'basepeh', 'basepeh',
                      'basepeh', 'basepeh', 'basepeh',
                      'basefft.amplitude', 'basefft.amplitude', 'basefft.amplitude']
        peakfields = ['peaklosses', 'peakcycles', 'chargepeak',
                      'peakparameters.form', 'peakparameters.crest',
                      'peakpeh', 'peakpeh', 'peakpeh',
                      'peakpeh', 'peakpeh', 'peakpeh',
                      'peakfft.amplitude', 'peakfft.amplitude', 'peakfft.amplitude']
        attrs = ['losses', 'cycles', 'charge',
                 'form', 'crest',
                 'powerpehlow', 'powerpehmed', 'powerpehhigh',
                 'energypehlow', 'energypehmed', 'energypehhigh',
                 'psdlow', 'psdmed', 'psdhigh']
        hooks = [scale_base_losses, scale, scale,
                 scale, scale,
                 powerpehlow, powerpehmed, powerpehhigh,
                 energypehlow, energypehmed, energypehhigh,
                 psdlow, psdmed, psdhigh]
        hookargs = [loss_scale, cycle_scale, charge_scale,
                    form_scale, crest_scale,
                    ppeh_scale[0], ppeh_scale[1], ppeh_scale[2],
                    epeh_scale[0], epeh_scale[1], epeh_scale[2],
                    psd_scale[0], psd_scale[1], psd_scale[2]]

        for attr, field, hook, arg in zip(attrs, basefields, hooks, hookargs):
            data = extract_data(hyb, field, copy_peak=False,
                                hook=hook, extrahookarg=arg)
            setattr(self.base, attr, SingularData(*data))

        # change first hook for peak losses, remaining hooks are equal
        hooks[0] = scale_peak_losses
        for attr, field, hook, arg in zip(attrs, peakfields, hooks, hookargs):
            data = extract_data(hyb, field, copy_peak=True,
                                hook=hook, extrahookarg=arg)
            setattr(self.peak, attr, SingularData(*data))

    @staticmethod
    def _get_scaling_factors(hyb):
        loss_scale = 1/hyb.single.losses*hyb.single.dim.energy
        cycle_scale = 1/hyb.single.cycles
        charge_scale = 1/hyb.single.dim.energy
        form_scale = 1/hyb.single.signalparameters.form
        crest_scale = 1/hyb.single.signalparameters.crest

        # power peh
        pweights = hyb.single.peh.pweights
        ppeh_scale = (float(np.sum(pweights[8:12])),
                      float(np.sum(pweights[2:8]) + np.sum(pweights[12:18])),
                      float(np.sum(pweights[0:2]) + np.sum(pweights[18:])))

        # energy peh
        eweights = hyb.single.peh.eweights
        epeh_scale = (float(np.sum(eweights[0:4])),
                      float(np.sum(eweights[4:16])),
                      float(np.sum(eweights[16:])))
        # TODO duplicate code with hooks at end of file

        # get integral of single fft
        psdvals = hyb.single.psd.amplitude
        psdint = np.cumsum(psdvals)
        # determine 90% limit
        psd09 = 0.90*psdint[-1]
        psdmaxindex = int(np.flatnonzero(np.array(psdint > psd09))[0] + 1)
        psdval = psdint[psdmaxindex]
        psd_1_3 = int(psdmaxindex*1/3 + 1)
        psd_2_3 = int(psdmaxindex*2/3 + 1)
        psd_scale = ((psdmaxindex,
                      np.sum(psdvals[0:psd_1_3-1]/psdval)),
                     (psdmaxindex,
                      np.sum(psdvals[psd_1_3:psd_2_3-1]/psdval)),
                     (psdmaxindex,
                      np.sum(psdvals[psd_2_3:psdmaxindex+1]/psdval)))

        return (loss_scale, cycle_scale, charge_scale, form_scale,
                crest_scale, ppeh_scale, epeh_scale, psd_scale)


class StorageResults:
    def __init__(self):
        self.losses = None
        self.form = None
        self.crest = None
        self.cycles = None
        self.powerpehlow = None
        self.powerpehmed = None
        self.powerpehhigh = None
        self.energypehlow = None
        self.energypehmed = None
        self.energypehhigh = None
        self.psdlow = None
        self.psdmed = None
        self.psdhigh = None
        self.charge = None


# ### Hooks
def psd(points, vals, start, end, singleendindexval):
    newvals = list()
    absend = singleendindexval[0]
    singleval = singleendindexval[1]
    for val in vals:
        allint = np.sum(val[0:absend]) if np.sum(val) else 1
        partint = np.sum(val[start:end])
        newvals.append(partint/allint/singleval)
    return points, newvals


def psdlow(points, vals, endindexval):
    start, end = 0, int(endindexval[0]/3)
    return psd(points, vals, start, end, endindexval)


def psdmed(points, vals, endindexval):
    start, end = int(endindexval[0]*1/3), int(endindexval[0]*2/3)
    return psd(points, vals, start, end, endindexval)


def psdhigh(points, vals, endindexval):
    start, end = int(endindexval[0]*2/3), endindexval[0] + 1
    return psd(points, vals, start, end, endindexval)


# PEH power assumes symmetric storages (P charge == P discharge)
# This will will not give wrong results if this is not the case,
# but eventually unpredicted ones. The PEHMap Class norms positive and
# negative power to the maximum occuring one (max(abs(p))), so exceeding
# power at one side will simply put into zero-bins
def powerpehhigh(points, vals, singleval):
    newvals = [float((np.sum(val.pweights[0:2]) +
                      np.sum(val.pweights[18:]))/singleval)
               for val in vals]
    return points, newvals


def powerpehmed(points, vals, singleval):
    newvals = [float((np.sum(val.pweights[2:8]) +
                      np.sum(val.pweights[12:18]))/singleval)
               for val in vals]
    return points, newvals


def powerpehlow(points, vals, singleval):
    newvals = [float(np.sum(val.pweights[8:12])/singleval) for val in vals]
    return points, newvals


def energypehlow(points, vals, singleval):
    newvals = [float(np.sum(val.eweights[0:4])/singleval) for val in vals]
    return points, newvals


def energypehmed(points, vals, singleval):
    newvals = [float(np.sum(val.eweights[4:16])/singleval) for val in vals]
    return points, newvals


def energypehhigh(points, vals, singleval):
    newvals = [float(np.sum(val.eweights[16:])/singleval) for val in vals]
    return points, newvals


def scale_base_losses(points, vals, scaleval):
    newvals = [float(val*scaleval/point[1]) if point[1] != 0 else 0
               for val, point in zip(vals, points)]
    return points, newvals


def scale_peak_losses(points, vals, scaleval):
    newvals = [float(val*scaleval/(1 - point[1])) if point[1] != 1 else 0
               for val, point in zip(vals, points)]
    return points, newvals


def scale(points, vals, scaleval):
    newvals = [float(val*scaleval) for val in vals]
    return points, newvals
