#!/usr/bin/env python3

import optess as oe
import sys
import glob


def merge_area(name):
    hyb = oe.HybridDia.load(name)
    areafiles = glob.glob('tmp_{}/{}_area_*'.format(name, name))
    for file in areafiles:
        hybarea = oe.HybridDia.load(file)
        for key, val in hybarea.area.items():
            hyb.area[key] = val
    hyb.save()


if __name__ == '__main__':
    NAME = sys.argv[1]
    merge_area(NAME)
