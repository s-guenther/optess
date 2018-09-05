#!/usr/bin/env python3

import optess as oe
import sys


def curve(filename, strategy, cut):
    """Loads the HybridDia Object specified in filename, performs single
    calculation, saves it."""
    hyb = oe.HybridDia.load(filename)
    hyb.calculate_point_at_curve(cut=cut, strategy=strategy)
    hyb.save(filename)


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    STRATEGY = sys.argv[2]
    CUT = float('0.' + sys.argv[3])
    curve(FILENAME, STRATEGY, CUT)
