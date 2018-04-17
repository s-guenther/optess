#!/usr/bin/env/python3
"""Defines data structure 'Signal'"""

from collections import namedtuple
from enum import Enum


Signal = namedtuple('Signal', 't x')
Storage = namedtuple('Storage', 'power efficiency discharge')


class Objective(Enum):
    power = 1
    energy = 2

# TODO implement type checking

# # protected property variables
# self._signal = None
# self._base = None
# self._peak = None
# self._objective = None
#
# @property
# def signal(self):
#     return self._signal
#
# @signal.setter
# def signal(self, val):
#     if isinstance(val, Signal):
#         self._signal = val
#     else:
#         self._signal = Signal(*val)
#
# @property
# def base(self):
#     return self._base
#
# @base.setter
# def base(self, val):
#     if isinstance(val, Storage):
#         self._base = val
#     else:
#         self._base = Storage(*val)
#
# @property
# def peak(self):
#     return self._peak
#
# @peak.setter
# def peak(self, val):
#     if isinstance(val, Storage):
#         self._peak = val
#     else:
#         self._peak = Storage(*val)
#
# @property
# def objective(self):
#     return self._objective
#
# @objective.setter
# def objective(self, val):
#     try:
#         if val.lower() is 'energy' or val.lower() is 'power':
#             self._objective = val.lower()
#         else:
#             raise ValueError("Objective must be 'energy' or 'power'")
#     except AttributeError:
#         TypeError('Objective must be string')
