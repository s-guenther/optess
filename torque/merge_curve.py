#!/usr/bin/env python3

import optess as oe
import sys
import glob


def abortedmsg(file):
    errmsg = '{}: Optimisation exceeded walltime or memory, nothing ' \
             'is added'
    print(errmsg.format(file))


def noresultmsg(file):
    errmsg = '{}: No Result found, nothing is added'
    print(errmsg.format(file))


def nofilesmsg():
    errmsg = 'No curve files found - single optimisation probably aborted'
    print(errmsg)


def merge_curve(name):
    hyb = oe.HybridDia.load(name)
    interfiles = glob.glob('tmp_{}/{}_curve_inter_*'.format(name, name))
    nointerfiles = glob.glob('tmp_{}/{}_curve_nointer_*'.format(name, name))
    if not interfiles and not nointerfiles:
        nofilesmsg()
    for file in interfiles:
        hybcut = oe.HybridDia.load(file)
        if not hybcut.inter:
            abortedmsg(file)
            continue
        for key, val in hybcut.inter.items():
            try:
                getattr(val, 'load_all_results')
            except AttributeError:
                noresultmsg(file)
                continue
            hyb.inter[key] = val
    for file in nointerfiles:
        hybcut = oe.HybridDia.load(file)
        if not hybcut.nointer:
            abortedmsg(file)
            continue
        for key, val in hybcut.nointer.items():
            try:
                getattr(val, 'load_all_results')
            except AttributeError:
                noresultmsg(file)
                continue
            hyb.nointer[key] = val
    # noinspection PyProtectedMember
    hyb._add_extreme_points()
    hyb.save()


if __name__ == '__main__':
    NAME = sys.argv[1]
    merge_curve(NAME)
