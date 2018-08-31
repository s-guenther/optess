#!/usr/bin/env bash

echo $(date) Starting single calculation >> ${NAME}.log

${MODULES}

cd ${WORKDIR}

python3 tmp_${NAME}/single.py ${NAME}.hyb >> ${NAME}.log

echo $(date) Finished single calculation >> ${NAME}.log
