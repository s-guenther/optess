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
