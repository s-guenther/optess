#!/usr/bin/env python3

import optess as oe
import sys
import glob


def abortedmsg(file):
    errmsg = '{}: Optimisation exceeded walltime or memory, or curve ' \
             'dependency was violated, nothing is added'
    print(errmsg.format(file))


def noresultmsg(file):
    errmsg = '{}: Found "NoResult" , nothing is added'
    print(errmsg.format(file))


def merge_area(name):
    hyb = oe.HybridDia.load(name)
    areafiles = glob.glob('tmp_{}/{}_area_*'.format(name, name))
    for file in areafiles:
        hybarea = oe.HybridDia.load(file)
        if not hybarea.area:
            abortedmsg(file)
            continue
        for key, val in hybarea.area.items():
            try:
                getattr(val, 'load_all_results')
            except AttributeError:
                noresultmsg(file)
                continue
            hyb.area[key] = val
    hyb.save()


if __name__ == '__main__':
    NAME = sys.argv[1]
    merge_area(NAME)
