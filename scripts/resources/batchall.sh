#!/bin/bash

NPOINTSLIST=("1000" "1500" "2200" \
             "3300" "4700" "6800" \
             "10000" "15000" "22000" \
             "33000" "47000" "68000" "100000")
CORESLIST=(8 8 8 \
          8 8 8 \
          8 8 8 \
          8 8 8 8)
WALLTIMELIST=(00:02:00 00:03:00 00:03:30 \
              00:06:00 00:08:00 00:12:00 \
              00:18:00 00:30:00 00:40:00 \
              00:60:00 01:20:00 02:00:00 03:30:00)
MEMLIST=(3GB 4GB 5GB \
         6GB 8GB 11GB \
         18GB 30GB 44GB \
         62GB 90GB 120GB 200GB)
PARTITIONLIST=(lena lena lena \
               lena lena lena \
               lena lena lena \
               lena smp smp smp)

export NPOINTS
export CORES
export WALLTIME
export MEM
export PARTITION
export NAME

DIRNAME=allsubs
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
    NAME="all${NPOINTSLIST[i]}"
    
    envsubst < /home/nhmcsgue/optess/scripts/resources/all.sh > $DIRNAME/$NAME.sh
done
    

