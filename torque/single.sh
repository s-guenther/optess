#!/bin/bash -login
cd ${WORKDIR}
echo $(date) Starting single calculation >> ${NAME}.log
module load ${MODULES}
python3 tmp_${NAME}/single.py ${NAME}.hyb >> ${NAME}.log
echo $(date) Finished single calculation >> ${NAME}.log
