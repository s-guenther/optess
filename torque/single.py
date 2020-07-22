#!/usr/bin/env python3

import optess as oe
import sys


def single(filename, singleenergy):
    hyb = oe.HybridDia.load(filename)
    hyb.calculate_single(singleenergy)
    hyb.save()


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    SINGLEENERGY = float(sys.argv[2])
    single(FILENAME, SINGLEENERGY)
