#!/usr/bin/env/python3
"""Defines data structure 'Signal'"""

from collections import namedtuple


Signal = namedtuple('Signal', 't x')
Storage = namedtuple('Storage', 'power efficiency discharge')
