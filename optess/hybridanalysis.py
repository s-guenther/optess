#!/usr/bin/env python3
"""Functions and methods which process the data of the hybridisation diagram
in a more complex way"""

from scipy.interpolate import griddata, interp1d
from scipy.integrate import quad
import numpy as np
from .utility import make_empty_axes
from matplotlib import pyplot as plt


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
