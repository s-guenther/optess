#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Starting area calculation \#${POINT} >> ${NAME}.log
cp ${WORKDIR}/${NAME}.hyb ${TMPDIR}/${NAME}_area_${POINT}.hyb
module load ${MODULES}
python3 ${TMPDIR}/area.py ${TMPDIR}/${NAME}_area_${POINT}.hyb \
    ${CURVES} ${POINT} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished area calculation \#${POINT} >> ${NAME}.log