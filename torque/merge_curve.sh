#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Start merging curves >> ${NAME}.log
module load ${MODULES}
python3 ${TMPDIR}/merge_curve.py ${NAME} >> ${WORKDIR}/${NAME}.log
echo $(date) Finished merging curves >> ${NAME}.log
