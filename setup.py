#!/usr/bin/env python3
# coding: utf-8

from setuptools import setup

setup(name='optess',
      version='0.1',
      description='Find an optimal energy storage system or hybrid energy '
                  'storage system for a certain load profile',
      url='http://github.com/s-guenther/optess',
      author='Sebastian Günther',
      author_email='sebastian.guenther@ifes.uni-hannover.de',
      license='GPLv3',
      packages=['optess'],
      install_requires=['matplotlib', 'pyomo', 'numpy', 'scipy', 'overload'],
      zip_safe=False)
