#!/usr/bin/env python3
"""Various factories to define data and optimization setups"""

import numpy as np
import scipy.interpolate as interp
from collections import namedtuple
from powersignal import Signal
from storage import Storage
from objective import Objective, Solver, Strategy


class UnknownDatatypeError(ValueError):
    pass


class UnknownStorageError(ValueError):
    pass


class UnknownObjectiveError(ValueError):
    pass


# ###
# ### Factories
# ###

# TODO replace strings with enums?


def datafactory(datatype, *args):
    """Builds a Signal representing a load profile as an input data for the
    optimisation.
    Currently implemented:
        datafactory('std', nsamples=64)
        datafactory('alt', nsamples=64)
    """
    if datatype == 'std':
        return _stdvals(*args)
    elif datatype == 'alt':
        return _altvals(*args)
    else:
        raise UnknownDatatypeError


def storagefactory(datatype):
    """Builds a Storage representing  as an input data for the optimisation.
    Currently implemented:
        storagefactory('<cut>.<loss>')
        where <cut> can be '025', '05', '075'
        and <loss> can be 'ideal', 'low', 'mid', 'high'
    Implementation hint: The input string is split at the point '.'. If new
    storages are added, the '<x>' and '<y>' strings must not contain a point
    character '.'. Also, a new implementation must contain this
    separating point.
    """
    sep = '.'
    pwr, prefs = datatype.split(sep)
    storageargs = []

    if pwr == '0':
        storageargs += [0]
    elif pwr == '025':
        storageargs += [0.25]
    elif pwr == '05':
        storageargs += [0.5]
    elif pwr == '075':
        storageargs += [0.75]
    elif pwr == '1':
        storageargs += [1]
    elif pwr == '2':
        storageargs += [2]
    else:
        raise UnknownStorageError

    # storageargs += [efficiency, selfdischarge]
    if prefs == 'ideal':
        storageargs += [1, 0]
    elif prefs == 'low':
        storageargs += [0.95, 0.01]
    elif prefs == 'med':
        storageargs += [0.9, 0.02]
    elif prefs == 'high':
        storageargs += [0.8, 0.05]
    else:
        raise UnknownStorageError

    return Storage(*storageargs)


def objectivefactory(datatype):
    """Builds a predefined objective as input data for te optimisation.
    Currently implemented:
        objectivefactory('std0-3')
    """
    if datatype == 'std0-3':
        return Objective('power', 3)
    else:
        raise UnknownObjectiveError


def hybridsetupfactory(setup, *args):
    """Builds a predefined setup for hybrid storageoptimisation, gathering
    object instantiation of the other factories (data, storage, objective).
    Currently implemented:
        optimsetupfactory('<data>.<storages>', cut='05')
        where <data> can take values of datafactory(),
        <storages can take values of <y> component of storagefactory()
        and cut can take values of <x> component of storagefactory().
        E.g.:
        optimsetupfactory('std.ideal')
        optimsetupfactory('alt.low', '025')

        Returns a namedtuple which cann be directly unpacked into OptimModel:
        model = OptimModel(*optsetup)
    """
    sep = '.'
    data, loss = setup.split(sep)
    if not args:
        cut = '05'
        cutbase = '1'
        cutpeak = '1'
    else:
        cut = args[0]
        cutbase = '1'
        cutpeak = '1'
    # if cut == '025':
    #     cutbase = '025'
    #     cutpeak = '075'
    # elif cut == '05':
    #     cutbase = '05'
    #     cutpeak = '05'
    # elif cut == '075':
    #     cutbase = '075'
    #     cutpeak = '025'
    # else:
    #     raise UnknownStorageError

    # This tuple can be unpacked directly into OptimModel
    HybridSetup = namedtuple('HybridSetup', 'signal base peak objective '
                                            'strategy solver name info')

    objective = 'std0-3'
    strategy = 'inter'
    solver = 'gurobi'
    name = 'Hybrid Storage Optimization ' \
           '{}.{}.{}.{}.{}'.format(data, cutbase, cutpeak, objective, strategy)
    info = None

    optsetup = HybridSetup(signal=datafactory(data, 110),
                           base=storagefactory(cutbase + '.' + loss),
                           peak=storagefactory(cutpeak + '.' + loss),
                           objective=objectivefactory(objective),
                           strategy=Strategy(strategy),
                           solver=Solver(solver),
                           name=name,
                           info=info)

    return optsetup


def singlesetupfactory(setup, *args):
    """Builds a predefined setup for single storageoptimisation, gathering
    object instantiation of the other factories (data, storage, objective).
    Currently implemented:
        optimsetupfactory('<data>.<storages>', power=1)
        where <data> can take values of datafactory(),
        <storages can take values of <y> component of storagefactory()
        and cut can take values of <x> component of storagefactory().
        E.g.:
        optimsetupfactory('std.ideal')
        optimsetupfactory('alt.low', '025')

        Returns a namedtuple which cann be directly unpacked into OptimModel:
        model = OptimModel(*optsetup)
    """
    sep = '.'
    data, loss = setup.split(sep)
    if not args:
        power = '2'
    else:
        power = args[0]

    # This tuple can be unpacked directly into OptimModel
    SingleSetup = namedtuple('SingleSetup', 'signal storage objective '
                                            'solver name info')

    objective = 'std0-3'
    solver = 'gurobi'
    name = 'Single Storage Optimization ' \
           '{}.{}.{}'.format(data, power, objective)
    info = None

    optsetup = SingleSetup(signal=datafactory(data, 110),
                           storage=storagefactory(power + '.' + loss),
                           objective=objectivefactory(objective),
                           solver=Solver(solver),
                           name=name,
                           info=info)

    return optsetup


# ###
# ### DataFactory Generating Functions
# ###

def _stdvals(nsamples=64):
    """Standard Testcase providing simple load profile"""
    x = range(0, 17)
    y = [3, 5, 3, 2, 3, 4, 3, 0, -1, 0, 3, 5, 3, -2, 3, 2, 2]
    # noinspection PyTypeChecker
    spl = interp.PchipInterpolator(x, y)
    xx = np.linspace((len(x)-1)/nsamples, len(x)-1, nsamples)
    yy = spl(xx)
    return Signal(xx, yy)


def _altvals(nsamples=64):
    """Alternative Testcase providing simple load profile"""
    x = range(23)
    y = [3, 5, 2, 5, 3, 5, 2, 5, 3, 5, 2, 5, 3, 1, 2, 3, 2, 1, 2, 2, 2, 2, 2]
    spl = interp.interp1d(x, y, 'linear')
    xx = np.linspace((len(x)-1)/nsamples, len(x)-1, nsamples)
    yy = spl(xx)
    return Signal(xx, yy)