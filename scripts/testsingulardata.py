#!/usr/bin/env python3

from optess.hybridanalysis import SingularData

# points = [(0, 0), (1e-12, 1-1e-12), (1, 1)]
points = [(0, 0), (0.5, 1-1e-12), (1, 1)]
# points = [(0, 0), (0.2, 0.4), (0.6, 0.8), (0.9, 0.97), (1, 1),
#           (0.3, 0.3), (0.7, 0.7)]
vals = [1, 1, 0]
# vals = [3, 6, 8, 12, 21, 4, 9]

sdata = SingularData(points, vals, datatype='test')

av = sdata.average
sdata.pplot()

dummybreakpoint = True
