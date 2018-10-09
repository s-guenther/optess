#!/usr/bin/env python3
"""Functions and methods which process the data of the hybridisation diagram
in a more complex way"""

from scipy.interpolate import griddata, interp1d
from scipy.integrate import quad, dblquad
import numpy as np
from .utility import make_empty_axes
from matplotlib import pyplot as plt


class SingularData:
    def __init__(self, points, vals, datatype=None, interp_method='linear'):
        self.points = list(points)
        self.vals = list(vals)
        self.type = datatype
        self._interpmethod = interp_method

    @property
    def hybridpotential(self):
        _, upper = self.get_lower_upper_bound()
        upperint, *err = quad(upper, 0, 1)
        if err[0] > 1e-6:
            print('Warning: Absolute Error >= 1e-6 (err = {})'.format(err))
        return 2*upperint - 1

    def interpolant(self, enorm, cut):
        """Returns callable object taking two argumens enorm, cut"""
        try:
            pointsinter = [(chi, mu) for chi, mu in zip(cut, enorm)]
        except TypeError:
            pointsinter = (cut, enorm)
        return griddata(self.points, self.vals, pointsinter,
                        method=self._interpmethod)

    @property
    def average(self):
        val, err = dblquad(self.interpolant,
                           0, 1,
                           *self.get_lower_upper_bound())
        if err > 1e-6:
            print('Warning: Absolute Error >= 1e-6 (err = {})'.format(err))
        return val/(self.hybridpotential/2)

    @staticmethod
    def _lower(cut):
        enorm = cut
        return enorm

    def _get_upper(self):
        pointdict = dict()
        for cut, enorm in self.points:
            if cut not in pointdict:
                pointdict[cut] = list()
            pointdict[cut].append(enorm)
        cuts, enorms = list(), list()
        for chi, mus in pointdict.items():
            cuts.append(chi)
            enorms.append(max(mus))
        cuts.sort()
        enorms.sort()
        return interp1d(cuts, enorms, kind='linear')

    def get_lower_upper_bound(self):
        """As a function of power cut chi in [0,1]"""
        return self._lower, self._get_upper()

    def is_point_in_area(self, point):
        """Point is (cut, enorm)"""
        cut, enorm = point
        if not 0 <= cut <= 1:
            return False
        lower, upper = self.get_lower_upper_bound()
        if not lower(cut) <= enorm <= upper(cut):
            return False
        else:
            return True

    def pplot(self, ax=None, intergrid=101, nbins=15, cmap='PuBu', **kwargs):
        y, x = [[*tup] for tup in zip(*self.points)]
        vals = list(self.vals)
        xunfiltered, yunfiltered = np.meshgrid(np.linspace(0, 1, intergrid),
                                               np.linspace(0, 1, intergrid))
        xunfiltered = xunfiltered.flatten()
        yunfiltered = yunfiltered.flatten()
        xfiltered = list()
        yfiltered = list()
        for xi, yi in zip(xunfiltered, yunfiltered):
            if self.is_point_in_area((yi, xi)):
                if not (yi, xi) in self.points:
                    xfiltered.append(xi)
                    yfiltered.append(yi)
        valsfiltered = self.interpolant(xfiltered, yfiltered)
        y += list(yfiltered)
        x += list(xfiltered)
        vals += list(valsfiltered)

        ax = ax if ax else make_empty_axes()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tricontour(x, y, vals, nbins, colors='k', linewidths=0.5, **kwargs)
        contour = ax.tricontourf(x, y, vals, nbins, cmap=cmap, **kwargs)
        fig = plt.gcf()
        fig.colorbar(contour, ax=ax)
