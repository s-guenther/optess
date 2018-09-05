#!/usr/bin/env python3

import optess as oe
import sys


def single(filename):
    hyb = oe.HybridDia.load(filename)
    hyb.calculate_single()
    hyb.save()


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    single(FILENAME)
