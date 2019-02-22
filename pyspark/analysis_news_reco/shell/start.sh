#!/usr/bin/env bash

REPORT_PATH=$1
NOCHAIN_SEED=$2


source $(dirname $0)/common.sh
ROOT=$(cd $(dirname $0) ; cd .. ; pwd)
SRC="${ROOT}/src"

num=$(ps x | grep search_spark.sh | grep -v "grep" | wc -l)
if [ $num -gt 3 ]; then
    FATAL "there is another procedure running, exit"
    exit 1
fi
echo "ROOT: ${ROOT}"
# spark-submit --master yarn-client --queue normal --num-executors 200 \
    --py-files ${SRC}/nochains_seed_analysis.py ${REPORT_PATH} ${NOCHAIN_SEED}