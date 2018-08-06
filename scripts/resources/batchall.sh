#!/bin/bash

NPOINTSLIST=(1000 1500 2200 \
             3300 4700 6800 \
             10000 15000 22000 \
             33000 47000 68000 100000)
CORELIST=(8 8 8 \
          8 8 8 \
          8 8 8 \
          8 8 8 8)
WALLTIMELIST=(00:02:00 00:03:00 00:03:30 \
              00:06:00 00:08:00 00:12:00 \
              00:18:00 00:30:00 00:40:00 \
              00:60:00 01:20:00 02:00:00 03:30:00)
MEMLIST=(3 4 5 \
         6 8 11 \
         18 30 44 \
         62 90 120 200)
PARTITIONLIST=(lena lena lena \
               lena lena lena \
               lena lena lena \
               lena smp smp smp)
NAMELIST=$NPOINTSLIST

export $NPOINTS
export $CORES
export $WALLTIME
export $MEM
export $PARTITION
export $NAME

DIRNAME=allsubs
if [-d $DIRNAME]
then
    mkdir $DIRNAME
fi

for ((i=0;i<=${#NPOINTSLIST[@]};i++))
do
    NPOINTS="{NPOINTSLIST[i]}"
    CORES="{CORESLIST[i]}"
    WALLTIME="{WALLTIMELIST[i]}"
    MEM="{MEMLIST[i]}"
    PARTITION="{PARTITIONLIST[i]}"
    NAME="all{NAMELIST[i]}"
    
    envsubslist < /home/nhmcsgue/optess/scripts/resources/all.sh > $DIRNAME/$NAME.sh
done
    

