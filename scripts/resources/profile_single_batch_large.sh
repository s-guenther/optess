#!/bin/bash

NPOINTSLIST=(22000 47000 100000 220000 \
             22000 47000 100000 220000 \
             22000 47000 100000 220000 \
             22000 47000 100000 220000)

CORESLIST=(1 1 1 1 \
           4 4 4 4 \
           8 8 8 8 \
           16 16 16 16)

WALLTIMELIST=(00:16:00 00:40:00 01:50:00 05:00:00 \
              00:08:00 00:36:00 01:50:00 05:00:00 \
              00:08:00 00:36:00 01:50:00 05:00:00 \
              00:08:00 00:36:00 01:50:00 05:00:00)

MEMLIST=(5GB 10GB 20GB 50GB \
         5GB 10GB 20GB 50GB \
         5GB 10GB 20GB 50GB \
         5GB 10GB 20GB 50GB)

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

DIRNAME=singlesubs_large
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
