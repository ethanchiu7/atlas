#!/bin/bash
source /home/datamining/.bash_profile

FILE_NAME=$0
# 相对路径
PROJECT_PATH=$(cd $(dirname $0); pwd)
# 绝对路径
PROJECT_PATH=$(readlink -f $0 | xargs dirname | xargs dirname)
echo "PROJECT_PATH : ${PROJECT_PATH}"
SHELL_PATH=${PROJECT_PATH}/shell

source ${PROJECT_PATH}/common/common.sh
source ${PROJECT_PATH}/common/common_hdfs.sh
# 需要修改配置: 执行环境 及 是否测试模式
source ${PROJECT_PATH}/conf/conf.sh

usage() {
    cat << EOT
Usage   :   ${FILE_NAME} [OPTION] ...
Options :
   chain_name
   -----------------------------------------------------------------------------
   chain_name:  |   translate_fun:     |  topN: |   hdfs_input:
   -----------  |   ---------------    |  ----  |  ---------------
Example:
EOT
}
# if [ ! $# -eq 4 ]; then usage; exit -1; fi

HDFS_INPUT_FOLDER=${1}

JOB_TIME=$(date +"%Y%m%d_%H%M%S")
FTR_DATE=$(date -d "-1 days" +"%Y%m%d")
EXPIRE_DATE=$(date -d "-7 days" "+%Y%m%d")

RESULT_TS=$(date "+%s")
expire_days=7

DELETE_DATE=$(date -d "-${expire_days} days" "+%Y-%m-%d")

LOCAL_OUTPUT=${PROJECT_PATH}/data
ENSURE_DIR ${LOCAL_OUTPUT}

ENSURE_DIR ${PROJECT_PATH}/log

HDFS_MERGED_SAMPLE_OUTPUT=""
HDFS_TRAIN_SAMPLE_OUTPUT=""
HDFS_TRAIN_SVM_SAMPLE_OUTPUT=""
HDFS_DELETE=""

function run_mr_merge_sample() {

    JOB_NAME="tuixing.zx.mr_merge_sample.${JOB_TIME}"

    HRM ${HDFS_MERGED_SAMPLE_OUTPUT}

    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming-2.6.0-cdh5.7.0.jar \
        -D mapred.reduce.tasks=100 \
        -D mapreduce.map.memory.mb=1024 \
        -D mapreduce.reduce.memory.mb=2048 \
        -D mapreduce.job.queuename=root.normal \
        -D mapreduce.job.name=${JOB_NAME} \
        -input ${HDFS_INPUT} \
        -output ${HDFS_MERGED_SAMPLE_OUTPUT} \
        -verbose \
        -mapper "/usr/local/bin/python2.7 map_sample_flow.py" \
        -reducer "/usr/local/bin/python2.7 reduce_sample_flow.py" \
        -file "${PROJECT_PATH}/conf/conf.py" \
        -file "${PROJECT_PATH}/src/utils.py" \
        -file "${PROJECT_PATH}/src/map_sample_flow.py" \
        -file "${PROJECT_PATH}/src/reduce_sample_flow.py"

    VERIFY_STATUS $? ${JOB_NAME}

    NOTICE "PROJECT_PATH: ${PROJECT_PATH}"
    NOTICE "HDFS_INPUT: ${HDFS_INPUT}"
    NOTICE "HDFS_MERGED_SAMPLE_OUTPUT: ${HDFS_MERGED_SAMPLE_OUTPUT}"

}

function run_mr_generate_train_sample() {

    JOB_NAME="tuixing.zx.mr_gene_fea.${JOB_TIME}"

    HRM ${HDFS_TRAIN_SAMPLE_OUTPUT}

    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming-2.6.0-cdh5.7.0.jar \
        -D mapred.reduce.tasks=100 \
        -D mapreduce.map.memory.mb=1024 \
        -D mapreduce.job.queuename=root.normal \
        -D mapreduce.job.name=${JOB_NAME} \
        -input ${HDFS_MERGED_SAMPLE_OUTPUT} \
        -output ${HDFS_TRAIN_SAMPLE_OUTPUT} \
        -verbose \
        -mapper "/usr/local/bin/python2.7 map_generate_feature.py" \
        -reducer cat \
        -file "${PROJECT_PATH}/conf/conf.py" \
        -file "${PROJECT_PATH}/src/utils.py" \
        -file "${PROJECT_PATH}/src/map_generate_feature.py" \
        -file "${PROJECT_PATH}/src/reduce_sample_flow.py"

    VERIFY_STATUS $? ${JOB_NAME}

    NOTICE "PROJECT_PATH: ${PROJECT_PATH}"
    NOTICE "HDFS_INPUT: ${HDFS_MERGED_SAMPLE_OUTPUT}"
    NOTICE "HDFS_TRAIN_SAMPLE_OUTPUT: ${HDFS_TRAIN_SAMPLE_OUTPUT}"

}

function run_mr_generate_svm_sample() {

    JOB_NAME="tuixing.zx.mr_svm_fea.${JOB_TIME}"

    HRM ${HDFS_TRAIN_SVM_SAMPLE_OUTPUT}

    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming-2.6.0-cdh5.7.0.jar \
        -D mapred.reduce.tasks=100 \
        -D mapreduce.map.memory.mb=1024 \
        -D mapreduce.job.queuename=root.normal \
        -D mapreduce.job.name=${JOB_NAME} \
        -input ${HDFS_MERGED_SAMPLE_OUTPUT} \
        -output ${HDFS_TRAIN_SVM_SAMPLE_OUTPUT} \
        -verbose \
        -mapper "/usr/local/bin/python2.7 map_generate_svm_feature.py" \
        -reducer cat \
        -file "${PROJECT_PATH}/conf/conf.py" \
        -file "${PROJECT_PATH}/src/utils.py" \
        -file "${PROJECT_PATH}/src/map_generate_svm_feature.py" \
        -file "${PROJECT_PATH}/src/reduce_sample_flow.py"

    VERIFY_STATUS $? ${JOB_NAME}

    NOTICE "PROJECT_PATH: ${PROJECT_PATH}"
    NOTICE "HDFS_INPUT: ${HDFS_MERGED_SAMPLE_OUTPUT}"
    NOTICE "HDFS_TRAIN_SVM_SAMPLE_OUTPUT: ${HDFS_TRAIN_SVM_SAMPLE_OUTPUT}"

}

function fetch_sample_file() {
    hdfs dfs -cat ${HDFS_TRAIN_SVM_SAMPLE_OUTPUT}/part* | cut -d ' ' -f 2- > ${LOCAL_OUTPUT}/train_sample_${JOB_TIME}.smv
}

function clean4project() {
    # clean HDFS
    HRM ${HDFS_DELETE}

    # clean data
    rm -f ${LOCAL_OUTPUT}/*${DELETE_DATE}*

}

run_mr_merge_sample

# run_mr_generate_train_sample

run_mr_generate_svm_sample

fetch_sample_file

clean4project

