#!/bin/bash

NPOINTSLIST=(1000 2200 4700 10000 \
             1000 2200 4700 10000 \
             1000 2200 4700 10000 \
             1000 2200 4700 10000)

CORESLIST=(1 1 1 1 \
           4 4 4 4 \
           8 8 8 8 \
           16 16 16 16)

WALLTIMELIST=(00:00:12 00:00:25 00:01:05 00:04:30 \
              00:00:12 00:00:16 00:00:50 00:03:10 \
              00:00:12 00:00:12 00:00:35 00:02:00 \
              00:00:12 00:00:12 00:00:22 00:01:40)

MEMLIST=(256MB 512MB 1024MB 2048MB \
         256MB 512MB 1024MB 2048MB \
         256MB 512MB 1024MB 2048MB \
         256MB 512MB 1024MB 2048MB)

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
    NAME=single_points_"$NPOINTS"_cores_"$CORES"_partition_"$PARTITION"

    echo Processing $NAME

    envsubst < /home/nhmcsgue/optess/scripts/resources/profile_single.sh > $DIRNAME/$NAME.sh
done

