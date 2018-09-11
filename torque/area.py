#!/usr/bin/env python3

import optess as oe
import sys


def area(filename, curves, pointnumber):
    hyb = oe.HybridDia.load(filename)
    points = hyb.get_points_in_area(curves)
    point = points[pointnumber]
    hyb.calculate_point_in_area(*point)
    hyb.save(filename)


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    CURVES = int(sys.argv[2])
    POINTNUMBER = int(sys.argv[3])
    area(FILENAME, CURVES, POINTNUMBER)