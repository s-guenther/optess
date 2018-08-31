"""This file stores some python and shell files as strings which can be
written to hdd and then executed"""

from collections import namedtuple
import xmltodict
from glob import glob
import os
import sys


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


JOIN_CURVE_SH = '''
Python file for
area calculation
'''[1:-1]


JOIN_CURVE_PY = '''
Python file for
area calculation
'''[1:-1]


JOIN_AREA_SH = '''
Python file for
area calculation
'''[1:-1]


JOIN_AREA_PY = '''
Python file for
area calculation
'''[1:-1]


CLEANUP_SH = '''
Python file for
area calculation
'''[1:-1]


def qsub(file, parameters=None, pbs=None):
    jobid = None
    return jobid


_Scale = namedtuple('_Scale', 'wt mem')
_Resources = namedtuple('_Resources', 'points time cpu ram')


class Torque:
    def __init__(self, hyb, setupfile=None, wt=1.0, mem=1.0):
        self.name = hyb.name
        self.npoints = len(hyb.signal)

        self.scale = _Scale(wt, mem)

        setup = self.read_setup(setupfile)
        self.workdir, self.modules, self.pbs, self.mail, self.resources = setup
        self.savedir = os.path.join(self.workdir, self.name)
        self.tmpdir = os.path.join(self.savedir, 'tmp_{}'.format(self.name))
        self.logdir = os.path.join(self.savedir, 'log_{}'.format(self.name))

        self.singleid = None
        self.interids = list()
        self.nointerids = list()
        self.areaids = list()
        self.utilityids = list()

        self.initialize(hyb)

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
        os.mkdir(self.savedir)
        os.mkdir(self.tmpdir)
        os.mkdir(self.logdir)
        hyb.save(os.path.join(self.workdir, '{}.hyb'.format(self.name)))
        filenames = ('single.sh', 'single.py', 'curve.sh', 'curve.py',
                     'area.sh', 'area.py', 'join_curve.sh', 'join_curve.py',
                     'join_area.sh', 'join_area.py', 'cleanup.sh')
        contents = (SINGLE_SH, SINGLE_PY, CURVE_SH, CURVE_PY, AREA_SH,
                    AREA_PY, JOIN_CURVE_SH, JOIN_CURVE_PY, JOIN_AREA_SH,
                    JOIN_AREA_PY, CLEANUP_SH)
        for filename, content in zip(filenames, contents):
            with open(os.path.join(tmpdir, filename), 'w') as file:
                file.write(content)

        logfilename = os.path.join(self.workdir, '{}.log'.format(self.name))
        with open(logfilename) as logfile:
            logfile.write('Initialized all directories\n')

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
            setup = xmltodict.parse(file.read())

        workdir = setup['workdir']
        modules = 'module load ' + '\nmodule load '.join(setup['module'])
        try:
            pbs = ' '.join(setup['pbs'])
        except 'KeyError':
            pbs = ''
        mail = setup['mail']
        resources = dict()
        for calc in ['single', 'curve', 'area']:
            resources[calc] = \
                _Resources(_string_to_numlist(setup[calc]['points']),
                           _string_to_numlist(setup[calc]['time']),
                           _string_to_numlist(setup[calc]['cpu']),
                           _string_to_numlist(setup[calc]['ram']))

        return workdir, modules, pbs, mail, resources


def _string_to_numlist(string):
    return [int(val) for val in string.split()]
