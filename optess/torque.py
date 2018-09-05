"""This file stores some python and shell files as strings which can be
written to hdd and then executed"""

from collections import namedtuple
from scipy.interpolate import interp1d
import xmltodict
from glob import glob
import os
from subprocess import check_output


class NoTorqueSetupFileFoundError(FileNotFoundError):
    """If raised if looked for a .torquesetup file, but none was found"""
    pass


class TorqueSetupIncompleteError:
    """Raised if the loaded .torquesetup file does not contain the required
    fields."""
    pass


SINGLE_SH = '''
#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Starting single calculation >> ${NAME}.log
module load ${MODULES}
python3 tmp_${NAME}/single.py ${NAME}.hyb >> ${NAME}.log
echo $(date) Finished single calculation >> ${NAME}.log
'''[1:-1]


SINGLE_PY = '''
#!/usr/bin/env python3

import optess as oe
import sys


def single(filename):
    """Loads the HybridDia Object specified in filename, performs single
    calculation, saves it."""
    hyb = oe.HybridDia.load(filename)
    hyb.calculate_single()
    hyb.save()


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    single(FILENAME)
'''[1:-1]


CURVE_SH = '''
#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Starting curve calculation at cut ${CUT} ${STRATEGY} >> \
    ${NAME}.log
cp ${WORKDIR}/${NAME}.hyb ${TMPDIR}/${NAME}_curve_${STRATEGY}_${CUT}.hyb
module load ${MODULES}
python3 ${TMPDIR}/curve.py ${NAME}_curve_${STRATEGY}_${CUT}.hyb \
    ${STRATEGY} ${CUT} >> ${WORKDIR}${NAME}.log
echo $(date) Finished curve calculation at cut ${CUT} ${STRATEGY} >> \
    ${NAME}.log
'''[1:-1]


CURVE_PY = '''
#!/usr/bin/env python3

import optess as oe
import sys


def single(filename, strategy, cut):
    """Loads the HybridDia Object specified in filename, performs single
    calculation, saves it."""
    hyb = oe.HybridDia.load(filename)
    hyb.calculate_point_at_curve(cut=cut, strategy=strategy)
    hyb.save(filename)


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    STRATEGY = sys.argv[2]
    CUT = float(sys.argv[3])
    single(FILENAME, STRATEGY, CUT)
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
Shell file for
join curve
'''[1:-1]


JOIN_CURVE_PY = '''
Python file for
join curve
'''[1:-1]


JOIN_AREA_SH = '''
Shell file for
join area
'''[1:-1]


JOIN_AREA_PY = '''
Python file for
join area
'''[1:-1]


CLEANUP_SH = '''
Shell file for
cleanup
'''[1:-1]


def qsub(file, parameters=None, pbs=None):
    """Submits the file 'file' with qsub, where the paramaters
    'parameters' are handed to the file and where the pbs directives
    'pbs' are used. 'parameters' is a string name:value pair dict, 'pbs' is a
    list of strings."""
    if parameters:
        paralist = []
        for para, val in parameters.items():
            paralist.append('{}="{}"'.format(para, val))
        parastring = '-v ' + ','.join(paralist)
    else:
        parastring = ''

    if pbs:
        pbsstring = ' '.join(pbs)
    else:
        pbsstring = ''

    command = ' '.join(['qsub', pbsstring, parastring, file])
    output = check_output(command, shell=True)
    jobid = output.decode('utf-8')[:-1]
    return jobid


_Scale = namedtuple('_Scale', 'wt mem')
_Resources = namedtuple('_Resources', 'points time cpu ram')


class Torque:
    def __init__(self, hyb, setupfile=None, wt=1.0, mem=1.0):
        self.name = hyb.name
        self.npoints = len(hyb.signal)

        self.scale = _Scale(wt, mem)

        self.workdir, self.architecture, self.modules, \
            self.mail, self.resources = self.read_setup(setupfile)
        self.savedir = os.path.join(self.workdir, self.name)
        self.tmpdir = os.path.join(self.workdir, 'tmp_{}'.format(self.name))
        self.logdir = os.path.join(self.workdir, 'log_{}'.format(self.name))

        self.singleid = None
        self.interids = list()
        self.nointerids = list()
        self.areaids = list()
        self.utilityids = list()

        self.initialize(hyb)

    def qsub(self, cuts, curves):
        self.singleid = self.qsub_single()
        for cut in cuts:
            self.interids.append(self.qsub_point_at_curve(cut, 'inter'))
        for cut in cuts:
            self.nointerids.append(self.qsub_point_at_curve(cut, 'nointer'))
        return True

    def qsub_single(self):
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir

        time, cores, ram = self.get_resources(self.npoints, 'single')
        arch = self.architecture
        nodes = 1

        pbs = list()
        pbs.append('-N {}_single'.format(self.name))
        pbs.append('-o {}/single.o'.format(self.logdir))
        pbs.append('-M {}'.format(self.mail))
        pbs.append('-m b')
        pbs.append('-j oe')
        pbs.append('-l nodes={}:{}:ppn={}'.format(nodes, arch, cores))
        pbs.append('-l mem={}'.format(ram))
        pbs.append('-l walltime={}'.format(time))

        jobid = qsub(os.path.join(self.tmpdir, 'single.sh'), paras, pbs)
        return jobid

    def qsub_point_at_curve(self, cut, strategy='inter'):
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir
        paras['TMPDIR'] = self.tmpdir
        paras['STRATEGY'] = strategy
        paras['CUT'] = str(cut)

        time, cores, ram = self.get_resources(self.npoints, 'single')
        arch = self.architecture
        nodes = 1

        pbs = list()
        pbs.append('-N {}_curve_{}_{}'.format(self.name, strategy, cut))
        pbs.append('-o {}/curve_{}_{}.o'.format(self.logdir, strategy, cut))
        pbs.append('-j oe')
        pbs.append('-l nodes={}:{}:ppn={}'.format(nodes, arch, cores))
        pbs.append('-l mem={}'.format(ram))
        pbs.append('-l walltime={}'.format(time))
        pbs.append('-W depend=afterok:{}'.format(self.singleid))

        jobid = qsub(os.path.join(self.tmpdir, 'curve.sh'), paras, pbs)
        return jobid

    def qsub_point_in_area(self, curves):
        jobid = None
        return jobid

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
        if not os.path.isdir(self.workdir):
            os.mkdir(self.workdir)
        if not os.path.isdir(self.savedir):
            os.mkdir(self.savedir)
        if not os.path.isdir(self.tmpdir):
            os.mkdir(self.tmpdir)
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)
        if os.path.isdir(self.name):
            os.rmdir(self.name)
        hyb.save(os.path.join(self.workdir, '{}.hyb'.format(self.name)))
        filenames = ('single.sh', 'single.py', 'curve.sh', 'curve.py',
                     'area.sh', 'area.py', 'join_curve.sh', 'join_curve.py',
                     'join_area.sh', 'join_area.py', 'cleanup.sh')
        contents = (SINGLE_SH, SINGLE_PY, CURVE_SH, CURVE_PY, AREA_SH,
                    AREA_PY, JOIN_CURVE_SH, JOIN_CURVE_PY, JOIN_AREA_SH,
                    JOIN_AREA_PY, CLEANUP_SH)
        for filename, content in zip(filenames, contents):
            with open(os.path.join(self.tmpdir, filename), 'w') as file:
                file.write(content)

        logfilename = os.path.join(self.workdir, '{}.log'.format(self.name))
        with open(logfilename, 'w') as logfile:
            logfile.write('Initialized all directories\n')

    def get_resources(self, n, calctype):
        points, times, cpus, rams = self.resources[calctype]
        cpu = str(int(interp1d(points, cpus, 'linear')(n)))
        ram = str(int(interp1d(points, rams, 'linear')(n))) + 'MB'
        ftime = int(interp1d(points, times, 'linear')(n))
        time = '{:02d}:{:02d}:{:02d}'.format(ftime//3600,
                                             (ftime - ftime//3600*3600)//60,
                                             ftime % 60)  # hours, min, sec
        return time, cpu, ram

    def status(self):
        pass

    def read_setup(self, setupfile=None):
        """If no setupfile provided, try reading the first .torquesetup
        file in current directory. Parses the xml document and writes
        structure as dict to self._torquesetup"""
        if not setupfile:
            try:
                setupfile = glob('*.torquesetup')[0]
            except IndexError:
                raise NoTorqueSetupFileFoundError

        with open(setupfile, 'rb') as file:
            setup = xmltodict.parse(file.read())['torquesetup']

        basedir = os.path.join(setup['workdir'])
        workdir = os.path.join(basedir, self.name)
        architecture = setup['architecture']
        modules = ' '.join(setup['module'])
        mail = setup['mail']
        resources = dict()
        for calc in ['single', 'curve', 'area']:
            resources[calc] = \
                _Resources(_string_to_numlist(setup[calc]['points']),
                           _string_to_numlist(setup[calc]['time']),
                           _string_to_numlist(setup[calc]['cpu']),
                           _string_to_numlist(setup[calc]['ram']))

        return workdir, architecture, modules, mail, resources


def _string_to_numlist(string):
    return [int(val) for val in string.split()]
