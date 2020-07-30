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
python3 tmp_${NAME}/single.py ${NAME}.hyb ${SINGLEENERGY} >> ${NAME}.log
echo $(date) Finished single calculation >> ${NAME}.log
'''[1:-1]


SINGLE_PY = '''
#!/usr/bin/env python3

import optess as oe
import sys


def single(filename, singleenergy):
    hyb = oe.HybridDia.load(filename)
    hyb.calculate_single(singleenergy)
    hyb.save()


if __name__ == '__main__':
    FILENAME = sys.argv[1]
    SINGLEENERGY = None if sys.argv[2] == 'None' else float(sys.argv[2])
    single(FILENAME, SINGLEENERGY)
'''[1:-1]


CURVE_SH = '''
#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Starting curve calculation at cut ${CUT} ${STRATEGY} >> \
    ${NAME}.log
cp ${WORKDIR}/${NAME}.hyb ${TMPDIR}/${NAME}_curve_${STRATEGY}_${CUT}.hyb
module load ${MODULES}
python3 ${TMPDIR}/curve.py ${TMPDIR}/${NAME}_curve_${STRATEGY}_${CUT}.hyb \
    ${STRATEGY} ${CUT} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished curve calculation at cut ${CUT} ${STRATEGY} >> \
    ${NAME}.log
'''[1:-1]


CURVE_PY = '''
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
'''[1:-1]


AREA_SH = '''
#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Starting area calculation \#${POINT} >> ${NAME}.log
cp ${WORKDIR}/${NAME}.hyb ${TMPDIR}/${NAME}_area_${POINT}.hyb
module load ${MODULES}
python3 ${TMPDIR}/area.py ${TMPDIR}/${NAME}_area_${POINT}.hyb \
    ${CURVES} ${POINT} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished area calculation \#${POINT} >> ${NAME}.log
'''[1:-1]


AREA_PY = '''
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
'''[1:-1]


MERGE_CURVE_SH = '''
#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Start merging curves >> ${NAME}.log
module load ${MODULES}
python3 ${TMPDIR}/merge_curve.py ${NAME} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished merging curves >> ${NAME}.log
'''[1:-1]


MERGE_CURVE_PY = '''
#!/usr/bin/env python3

import optess as oe
import sys
import glob


def abortedmsg(file):
    errmsg = '{}: Optimisation exceeded walltime or memory, nothing ' \
             'is added'
    print(errmsg.format(file))


def noresultmsg(file):
    errmsg = '{}: No Result found, nothing is added'
    print(errmsg.format(file))


def nofilesmsg():
    errmsg = 'No curve files found - single optimisation probably aborted'
    print(errmsg)


def merge_curve(name):
    hyb = oe.HybridDia.load(name)
    interfiles = glob.glob('tmp_{}/{}_curve_inter_*'.format(name, name))
    nointerfiles = glob.glob('tmp_{}/{}_curve_nointer_*'.format(name, name))
    if not interfiles and not nointerfiles:
        nofilesmsg()
    for file in interfiles:
        hybcut = oe.HybridDia.load(file)
        if not hybcut.inter:
            abortedmsg(file)
            continue
        for key, val in hybcut.inter.items():
            try:
                getattr(val, 'load_all_results')
            except AttributeError:
                noresultmsg(file)
                continue
            hyb.inter[key] = val
    for file in nointerfiles:
        hybcut = oe.HybridDia.load(file)
        if not hybcut.nointer:
            abortedmsg(file)
            continue
        for key, val in hybcut.nointer.items():
            try:
                getattr(val, 'load_all_results')
            except AttributeError:
                noresultmsg(file)
                continue
            hyb.nointer[key] = val
    # noinspection PyProtectedMember
    hyb._add_extreme_points()
    hyb.save()


if __name__ == '__main__':
    NAME = sys.argv[1]
    merge_curve(NAME)
'''[1:-1]


MERGE_AREA_SH = '''
#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Start merging area >> ${NAME}.log
module load ${MODULES}
python3 ${TMPDIR}/merge_area.py ${NAME} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished merging area >> ${NAME}.log
echo Cleaning up... >> ${NAME}.log
rm -rf ${TMPDIR}
echo "All done" >> ${NAME}.log
'''[1:-1]


MERGE_AREA_PY = '''
#!/usr/bin/env python3

import optess as oe
import sys
import glob


def abortedmsg(file):
    errmsg = '{}: Optimisation exceeded walltime or memory, or curve ' \
             'dependency was violated, nothing is added'
    print(errmsg.format(file))


def noresultmsg(file):
    errmsg = '{}: Found "NoResult" , nothing is added'
    print(errmsg.format(file))


def merge_area(name):
    hyb = oe.HybridDia.load(name)
    areafiles = glob.glob('tmp_{}/{}_area_*'.format(name, name))
    for file in areafiles:
        hybarea = oe.HybridDia.load(file)
        if not hybarea.area:
            abortedmsg(file)
            continue
        for key, val in hybarea.area.items():
            try:
                getattr(val, 'load_all_results')
            except AttributeError:
                noresultmsg(file)
                continue
            hyb.area[key] = val
    hyb.save()


if __name__ == '__main__':
    NAME = sys.argv[1]
    merge_area(NAME)
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

    def qsub(self, cuts, curves, singleenergy=None):
        self.singleid = self.qsub_single(singleenergy)
        for cut in cuts:
            self.interids.append(self.qsub_point_at_curve(cut, 'inter'))
        for cut in cuts:
            self.nointerids.append(self.qsub_point_at_curve(cut, 'nointer'))
        self.utilityids.append(self.qsub_merge_curve())
        npoints = len(cuts)*curves
        for pointnumber in range(npoints):
            self.areaids.append(self.qsub_point_in_area(curves, pointnumber))
        self.utilityids.append(self.qsub_merge_area())
        return True

    def qsub_single(self, singleenergy):
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir
        paras['SINGLEENERGY'] = str(singleenergy)

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
        # TODO do not submit cut=0 or cut=1
        cutstr = str(cut)[2:]
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir
        paras['TMPDIR'] = self.tmpdir
        paras['STRATEGY'] = strategy
        paras['CUT'] = cutstr

        time, cores, ram = self.get_resources(self.npoints, strategy)
        arch = self.architecture
        nodes = 1

        pbs = list()
        pbs.append('-N {}_curve_{}_{}'.format(self.name, strategy, cutstr))
        pbs.append('-o {}/curve_{}_{}.o'.format(self.logdir, strategy, cutstr))
        pbs.append('-j oe')
        pbs.append('-l nodes={}:{}:ppn={}'.format(nodes, arch, cores))
        pbs.append('-l mem={}'.format(ram))
        pbs.append('-l walltime={}'.format(time))
        pbs.append('-W depend=afterok:{}'.format(self.singleid))

        jobid = qsub(os.path.join(self.tmpdir, 'curve.sh'), paras, pbs)
        return jobid

    def qsub_point_in_area(self, curves, pointnumber):
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir
        paras['TMPDIR'] = self.tmpdir
        paras['CURVES'] = curves
        paras['POINT'] = pointnumber

        time, cores, ram = self.get_resources(self.npoints, 'area')
        arch = self.architecture
        nodes = 1

        pbs = list()
        pbs.append('-N {}_area_{:02d}'.format(self.name, pointnumber))
        pbs.append('-o {}/area_{:02d}.o'.format(self.logdir, pointnumber))
        pbs.append('-j oe')
        pbs.append('-l nodes={}:{}:ppn={}'.format(nodes, arch, cores))
        pbs.append('-l mem={}'.format(ram))
        pbs.append('-l walltime={}'.format(time))
        pbs.append('-W depend=afterok:{}'.format(self.utilityids[0]))

        jobid = qsub(os.path.join(self.tmpdir, 'area.sh'), paras, pbs)
        return jobid

    def qsub_merge_curve(self):
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir
        paras['TMPDIR'] = self.tmpdir

        time, cores, ram = '00:00:30', 1, '1GB'
        nodes, architecture = 1, 'haswell'

        pbs = list()
        pbs.append('-N {}_merge_curve'.format(self.name))
        pbs.append('-o {}/merge_curve.o'.format(self.logdir))
        pbs.append('-j oe')
        pbs.append('-l nodes={}:{}:ppn={}'.format(nodes, architecture, cores))
        pbs.append('-l mem={}'.format(ram))
        pbs.append('-l walltime={}'.format(time))
        pbs.append('-W depend=afterany:{}'.format(':'.join(self.interids +
                                                           self.nointerids)))

        jobid = qsub(os.path.join(self.tmpdir, 'merge_curve.sh'), paras, pbs)
        return jobid

    def qsub_merge_area(self):
        paras = dict()
        paras['MODULES'] = self.modules
        paras['NAME'] = self.name
        paras['WORKDIR'] = self.workdir
        paras['TMPDIR'] = self.tmpdir

        time, cores, ram = '00:00:30', 1, '1GB'
        nodes, architecture = 1, 'haswell'

        pbs = list()
        pbs.append('-N {}_merge_area'.format(self.name))
        pbs.append('-M {}'.format(self.mail))
        pbs.append('-m ae')
        pbs.append('-o {}/merge_area.o'.format(self.logdir))
        pbs.append('-j oe')
        pbs.append('-l nodes={}:{}:ppn={}'.format(nodes, architecture, cores))
        pbs.append('-l mem={}'.format(ram))
        pbs.append('-l walltime={}'.format(time))
        pbs.append('-W depend=afterany:{}'.format(':'.join(self.areaids)))

        jobid = qsub(os.path.join(self.tmpdir, 'merge_area.sh'), paras, pbs)
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
                     'area.sh', 'area.py', 'merge_curve.sh', 'merge_curve.py',
                     'merge_area.sh', 'merge_area.py')
        contents = (SINGLE_SH, SINGLE_PY, CURVE_SH, CURVE_PY, AREA_SH,
                    AREA_PY, MERGE_CURVE_SH, MERGE_CURVE_PY, MERGE_AREA_SH,
                    MERGE_AREA_PY)
        for filename, content in zip(filenames, contents):
            with open(os.path.join(self.tmpdir, filename), 'w') as file:
                file.write(content)

        logfilename = os.path.join(self.workdir, '{}.log'.format(self.name))
        with open(logfilename, 'w') as logfile:
            logfile.write('Initialized all directories\n')

    def get_resources(self, n, calctype):
        wt, mem = self.scale
        points, times, cpus, rams = self.resources[calctype]
        cpu = str(int(interp1d(points, cpus, 'linear')(n)))
        ram = str(int(mem*interp1d(points, rams, 'linear')(n))) + 'MB'
        ftime = int(wt*interp1d(points, times, 'linear')(n))
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
        for calc in ['single', 'inter', 'nointer', 'area']:
            resources[calc] = \
                _Resources(_string_to_numlist(setup[calc]['points']),
                           _string_to_numlist(setup[calc]['time']),
                           _string_to_numlist(setup[calc]['cpu']),
                           _string_to_numlist(setup[calc]['ram']))

        return workdir, architecture, modules, mail, resources


def _string_to_numlist(string):
    return [int(val) for val in string.split()]
