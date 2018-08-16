#!/bin/bash

NPOINTSLIST=(1000 2200 4700 10000 \
             1000 2200 4700 10000 \
             1000 2200 4700 10000 \
             1000 2200 4700 10000)

CORESLIST=(1 2 4 8 \
           1 2 4 8 \
           1 2 4 8 \
           1 2 4 8)

WALLTIMELIST=(00:00:05 00:00:07 00:00:12 00:00:22 \
              00:00:05 00:00:07 00:00:12 00:00:22 \
              00:00:05 00:00:07 00:00:12 00:00:22 \
              00:00:05 00:00:07 00:00:12 00:00:22)

MEMLIST=(350MB 650MB 1350MB 2600MB \
         350MB 650MB 1350MB 2600MB \
         350MB 650MB 1350MB 2600MB \
         350MB 650MB 1350MB 2600MB)

PARTITIONLIST=(lena lena lena lena \
               lena lena lena lena \
               lena lena lena lena \
               lena lena lena lena)

export NPOINTS
export CORES
export WALLTIME
export MEM
export PARTITION
export NAME

DIRNAME=singlesubs
if [ ! -d $DIRNAME ]
then
    mkdir $DIRNAME
fi

for ((i=0;i<${#NPOINTSLIST[@]};i++))
do
    NPOINTS="${NPOINTSLIST[i]}"
    CORES="${CORESLIST[i]}"
    WALLTIME="${WALLTIMELIST[i]}"
    MEM="${MEMLIST[i]}"
    PARTITION="${PARTITIONLIST[i]}"
    NAME="single_points_$NPOINTS_cores_$CORES_partition_$PARTITION"

    envsubst < /home/nhmcsgue/optess/scripts/resources/profile_single.sh > $DIRNAME/$NAME.sh
done

