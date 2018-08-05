#!/usr/bin/env python3
"""
Creates a random signal with 3300 points and calculates the hybridisation
area
"""

import timeit
import os

import optess as oe

npoints = 3300

dia = oe.HybridDia.load('/bigwork/nhmcsgue/large/first{}'.format(npoints))

dia.name = 'second{}'.format(npoints)
os.mkdir('second{}'.format(npoints))

start = timeit.default_timer()
dia.calculate_area((11, 11))
end = timeit.default_timer() - start
print('Area Calculation went for {} seconds'.format(end))
print('Calculated points: {}'.format(len(dia.area.keys())))

dia.save()
