#!/bin/bash
# this should be on sf-iflow-s67
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
    analysis seed increase effect
    必须指定具体日期作为本次分析任务之参数
Options :
    date    the date to be analysised
Example:
    ${FILE_NAME} 2019-01-01
EOT
}
# if [ $# -eq 0 ]; then usage; exit -1; fi

RESULT_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_TS=$(date "+%s")
EXPIRE_DAYS=1

# -------

NOTICE "DOC: https://yuque.antfin-inc.com/ibd_feed/algorithms/dk6eym"

ENSURE_DIR ${PROJECT_PATH}/log
CHAIN_NAME="kRealtimeContext"
JOB_NAME="tuixing.zx.${CHAIN_NAME}2pangu"
JOB_HOME="job_home"

HDFS_INPUT="zhengfan/realtime_context/output_path"
HDFS_OUTPUT="${HDFS_PROJECT_ROOT}/${CHAIN_NAME}/${RESULT_TIME}"
LOCAL_OUTPUT="${PROJECT_PATH}/data"
ENSURE_DIR ${LOCAL_OUTPUT}
RESULT="${CHAIN_NAME}.multilang.${RESULT_TS}"

function run_mr() {

    # archive files
    declare -a mr_file_arr=("${PROJECT_PATH_kRealtimeContext}/similar/clk.d" \
                            "${PROJECT_PATH_kRealtimeContext}/similar/fine.d" \
                            "${PROJECT_PATH_kRealtimeContext}/similar/item_tm.d" \
                            "${PROJECT_PATH}/translator/utils.py" \
                            "${PROJECT_PATH}/translator/interpreter.py" \
                            "${PROJECT_PATH}/translator/kRealtimeContextTrasnlator.py")

    # ARCHIVE_FILES_TO_HDFS "${mr_file_arr[@]}" "${JOB_NAME}.pkg.tar" "${HDFS_MR_PKG}"

    HRM ${HDFS_OUTPUT}

    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming-2.6.0-cdh5.7.0.jar \
        -D mapred.reduce.tasks=1 \
        -D mapreduce.map.memory.mb=1024 \
        -D mapreduce.job.queuename=root.normal \
        -D mapreduce.job.name=${JOB_NAME} \
        -input ${HDFS_INPUT} \
        -output ${HDFS_OUTPUT} \
        -verbose \
        -mapper "/usr/local/bin/python2.7 kRealtimeContextTrasnlator.py --clk clk.d --fine fine.d --langmap item_tm.d" \
        -reducer cat \
        -file "${PROJECT_PATH_kRealtimeContext}/similar/clk.d" \
        -file "${PROJECT_PATH_kRealtimeContext}/similar/fine.d" \
        -file "${PROJECT_PATH_kRealtimeContext}/similar/item_tm.d" \
        -file "${PROJECT_PATH}/translator/utils.py" \
        -file "${PROJECT_PATH}/translator/interpreter.py" \
        -file "${PROJECT_PATH}/translator/kRealtimeContextTrasnlator.py"

    # -mapper "/usr/local/bin/python2.7 ${JOB_HOME}/kRealtimeContextTrasnlator.py --clk ${JOB_HOME}/clk.d --fine ${JOB_HOME}/fine.d --langmap ${JOB_HOME}/item_tm.d" \

    VERIFY_STATUS $? ${JOB_NAME}

    NOTICE "CHAIN_NAME: ${JOB_NAME}"
    NOTICE "PROJECT_PATH: ${PROJECT_PATH}"
    NOTICE "HDFS_INPUT: ${HDFS_INPUT}"
    NOTICE "HDFS_OUTPUT: ${HDFS_OUTPUT}"

}

function remove_pangu() {

    cd ${PROJECT_PATH}/log

    HGET ${HDFS_OUTPUT}/part* "${LOCAL_OUTPUT}/${RESULT}"

    # for bug of pangu
    ENSURE_DIR "__chain_info_dir_DO_NOT_DELETE__"

    fs_util_file=$(which fs_util | head -n1 | awk -F '=' '{print $2}' | sed $'s/\'//g')

    NOTICE "run chain_merge_and_sample.py ..."
    NOTICE "This is for bug of pangu !!!"
    NOTICE "python2.7 ${PROJECT_PATH}/translator/chain_merge_and_sample.py ${fs_util_file} ${LOCAL_OUTPUT}/${RESULT} ${PANGU_kRealtimeContext}"
    python2.7 ${PROJECT_PATH}/translator/chain_merge_and_sample.py ${fs_util_file} ${LOCAL_OUTPUT}/${RESULT} ${PANGU_kRealtimeContext}

#    NOTICE "cp ${LOCAL_OUTPUT}/${RESULT} -> pangu: ${PANGU_TMP}"
#    fs_util cp "${LOCAL_OUTPUT}/${RESULT}" ${PANGU_TMP}/${RESULT}
#
#    NOTICE "mv ${PANGU_TMP}/${RESULT} -> ${PANGU_kRealtimeContext}/${RESULT}"
#    fs_util mv ${PANGU_TMP}/${RESULT} ${PANGU_kRealtimeContext}/${RESULT}

    VERIFY_STATUS $? 'fs_util_mv'
}

function clean4project() {
    # backup
    NOTICE "mv ${LOCAL_OUTPUT}/${RESULT} ${LOCAL_OUTPUT}/${RESULT}.his"
    mv ${LOCAL_OUTPUT}/${RESULT} ${LOCAL_OUTPUT}/${RESULT}.his

    # clean data
    find ${LOCAL_OUTPUT} -name "*his" -mtime ${EXPIRE_DAYS} -delete -or -name "*txt" -mtime ${EXPIRE_DAYS} -delete
    # clean log
    find ${PROJECT_PATH}/log -name "*log" -mtime ${EXPIRE_DAYS} -delete -or -name "*.txt" -mtime ${EXPIRE_DAYS} -delete
}

run_mr

remove_pangu

clean4project