#!/bin/bash -login

#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:10:00
#PBS -l mem=6GB
#PBS -N second1500
#PBS -M sebastian.guenther@ifes.uni-hannover.de
#PBS -m abe

module load GCC/4.9.3-2.25 OpenMPI/1.10.2
module load Python/3.4.3
module load numpy/1.10.1-Python-3.4.3
module load scipy/0.17.1-Python-3.4.3
module load Gurobi/7.5.2

cd $BIGWORK/large

python3 /home/nhmcsgue/optess/scripts/resources/second1500.py
