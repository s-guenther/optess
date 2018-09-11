#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Start merging area >> ${NAME}.log
module load ${MODULES}
python3 ${TMPDIR}/merge_area.py ${NAME} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished merging area >> ${NAME}.log
echo Cleaning up... >> ${NAME}.log
rm -rf ${TMPDIR}
echo "All done" >> ${NAME}.log
