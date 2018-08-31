"""This file stores some python and shell files as strings which can be
written to hdd and then executed"""

from collections import namedtuple
import xmltodict
from glob import glob
import os


class NoTorqueSetupFileFoundError(FileNotFoundError):
    """If raised if looked for a .torquesetup file, but none was found"""
    pass


class TorqueSetupIncompleteError:
    """Raised if the loaded .torquesetup file does not contain the required
    fields."""
    pass


SINGLE_SH = '''
Shell file for single
calculation
'''[1:-1]


SINGLE_PY = '''
Python file for
single calculation
'''[1:-1]

CURVE_SH = '''
Shell file for curve
calculation
'''[1:-1]


CURVE_PY = '''
Python file for
curve calculation
'''[1:-1]


AREA_SH = '''
Shell file for area
calculation
'''[1:-1]


AREA_PY = '''
Python file for
area calculation
'''[1:-1]


def qsub(file, parameters=None, pbs=None):
    jobid = None
    return jobid


_Scale = namedtuple('_Scale', 'wt mem')


class Torque:
    def __init__(self, hyb, setupfile=None, wt=1.0, mem=1.0):
        self.name = hyb.name
        self.npoints = len(hyb.signal)
        self.setup = self.read_setup(setupfile)
        self.scale = _Scale(wt, mem)
        self.singleid = None
        self.interids = list()
        self.nointerids = list()
        self.areaids = list()
        self.utilityids = list()

        self.initialize(self, hyb)

    def qsub(self, cuts, curves):
        # do all
        return True

    def qsub_single(self):
        jobid = None
        return jobid

    def qsub_point_at_curve(self, cut, strategy='inter'):
        jobid = None
        return jobid

    def qsub_point_in_area(self, curves):
        jobid = None
        return jobids

    def qsub_join_curve(self):
        jobid = None
        return jobid

    def qsub_join_area(self):
        jobid = None
        return jobid

    def qsub_cleanup(self):
        jobid = None
        return jobid

    def initialize(self, hyb):
        # TODO change path to setuppath
        os.mkdir(self.name)
        tmpdir = 'tmp_{}'.format(self.name)
        os.mkdir(tmpdir)
        hyb.save()
        filenames = ('single.sh', 'single.py', 'curve.sh', 'curve.py',
                     'area.sh', 'area.py')
        contents = (SINGLE_SH, SINGLE_PY, CURVE_SH, CURVE_PY, AREA_SH, AREA_PY)
        for filename, content in zip(filenames, contents):
            with open(os.path.join(tmpdir, filename), 'w') as file:
                file.write(content)

    def status(self):
        pass

    @staticmethod
    def read_setup(setupfile=None):
        """If no setupfile provided, try reading the first .torquesetup
        file in current directory. Parses the xml document and writes
        structure as dict to self._torquesetup"""
        if not setupfile:
            try:
                setupfile = glob('*.torquesetup')[0]
            except IndexError:
                raise NoTorqueSetupFileFoundError

        with open(setupfile, 'wb') as file:
            torquesetup = xmltodict.parse(file.read())

        torquekeys = ('workdir', 'modules', 'single', 'curve', 'area')
        valid = all(key in torquesetup for key in torquekeys)
        if not valid:
            raise TorqueSetupIncompleteError
        return setup
