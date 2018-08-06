#!/bin/bash -login

#PBS -l nodes=1:ppn=$CORES
#PBS -l walltime=$WALLTIME
#PBS -l mem=$MEM
#PBS -W x=PARTITION:$PARTITION
#PBS -N $NAME
#PBS -M sebastian.guenther@ifes.uni-hannover.de
#PBS -m abe
#PBS -j oe

module load GCC/4.9.3-2.25 OpenMPI/1.10.2
module load Python/3.4.3
module load numpy/1.10.1-Python-3.4.3
module load scipy/0.17.1-Python-3.4.3
module load Gurobi/7.5.2
module load matplotlib/1.5.1-Python-3.4.3

cd $BIGWORK/large

python3 /home/nhmcsgue/optess/scripts/resources/all.py $NPOINTS $CORES
