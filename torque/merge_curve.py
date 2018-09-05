#!/usr/bin/env python3

import optess as oe
import sys
import glob


def merge_curve(name):
    hyb = oe.HybridDia.load(name)
    interfiles = glob.glob('tmp_{}/{}_curve_inter_*'.format(name, name))
    nointerfiles = glob.glob('tmp_{}/{}_curve_nointer_*'.format(name, name))
    for file in interfiles:
        hybcut = oe.HybridDia.load(file)
        for key, val in hybcut.inter.items():
            hyb.inter[key] = val
    for file in nointerfiles:
        hybcut = oe.HybridDia.load(file)
        for key, val in hybcut.nointer.items():
            hyb.nointer[key] = val
    hyb.save()


if __name__ == '__main__':
    NAME = sys.argv[1]
    merge_curve(NAME)