#!/bin/bash -login

#PBS -l nodes=1:ppn=12
#PBS -l walltime=00:08:00
#PBS -l mem=14GB
#PBS -N first2200
#PBS -M sebastian.guenther@ifes.uni-hannover.de
#PBS -m abe

module load GCC/4.9.3-2.25 OpenMPI/1.10.2
module load Python/3.4.3
module load numpy/1.10.1-Python-3.4.3
module load scipy/0.17.1-Python-3.4.3
module load Gurobi/7.5.2

cd $BIGWORK/large

python3 /home/nhmcsgue/optess/first2200.py
