#!/bin/bash
source /home/datamining/.bash_profile
# ------------------

# 绝对路径
PROJECT_PATH=$(readlink -f $0 | xargs dirname | xargs dirname)
echo "PROJECT_PATH : ${PROJECT_PATH}"
SHELL_PATH=${PROJECT_PATH}/shell

source ${PROJECT_PATH}/common/common.sh
source ${PROJECT_PATH}/common/common_hdfs.sh
# 需要修改配置: 执行环境 及 是否测试模式
source ${PROJECT_PATH}/conf/conf.sh

RESULT_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_TS=$(date "+%s")
EXPIRE_DAYS=1

ENSURE_DIR ${PROJECT_PATH}/log
LANGUAGE=$1
CHAIN_NAME="kHighConsume"
JOB_NAME="tuixing.zx.${CHAIN_NAME}.${LANGUAGE}2pangu"
JOB_HOME="job_home"


HDFS_INPUT="/user/datamining/chenwei/work/high_consume/vector_sim_ret/${LANGUAGE}/output_path"
HDFS_OUTPUT="${HDFS_PROJECT_ROOT}/${CHAIN_NAME}/${RESULT_TIME}.${LANGUAGE}"
LOCAL_OUTPUT="${PROJECT_PATH}/data"
ENSURE_DIR ${LOCAL_OUTPUT}
RESULT="${CHAIN_NAME}.${LANGUAGE}.${RESULT_TS}"

# ------------------

ORI_PATH="/home/datamining/chenweiwei.cw/workspace/high_consume/build_chain/send_tair"

function run_mr() {
    # archive files
    declare -a mr_file_arr=("${PROJECT_PATH}/translator/kHighConsumeTranslator.py" \
                            "${ORI_PATH}/clk_id_file.${LANGUAGE}" \
                            "${ORI_PATH}/consume_id_file.${LANGUAGE}" \
                            "${ORI_PATH}/itemtype2itemlist")

    ARCHIVE_FILES_TO_HDFS "${mr_file_arr[@]}" "${JOB_NAME}" "${HDFS_DATAMINING}/${HDFS_MR_PKG}"
    HRM ${HDFS_OUTPUT}

    hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming-2.6.0-cdh5.7.0.jar \
          -archives "${HDFS_DATAMINING}/${HDFS_MR_PKG}/${JOB_NAME}.tar#src" \
          -Dmapreduce.job.queuename=root.normal \
          -Dmapreduce.job.name=chenweiwei.cw-build_chain  \
          -Dmapreduce.job.priority=HIGH \
          -Dmapreduce.map.memory.mb=4096  \
          -Dmapred.reduce.tasks=1 \
          -input "${HDFS_INPUT}" \
          -output "${HDFS_OUTPUT}" \
          -verbose \
          -mapper "/usr/local/bin/python2.7 src/kHighConsumeTranslator.py src/clk_id_file.${LANGUAGE}  src/consume_id_file.${LANGUAGE} src/itemtype2itemlist  ${LANGUAGE}"
}

function remove_pangu() {

    cd ${PROJECT_PATH}/log

    HGET ${HDFS_OUTPUT}/part* "${LOCAL_OUTPUT}/${RESULT}"

    # for bug of pangu
    ENSURE_DIR "__chain_info_dir_DO_NOT_DELETE__"

    fs_util_file=$(which fs_util | head -n1 | awk -F '=' '{print $2}' | sed $'s/\'//g')

    NOTICE "run chain_merge_and_sample.py ..."
    NOTICE "This is for bug of pangu !!!"
    NOTICE "python2.7 ${PROJECT_PATH}/translator/chain_merge_and_sample.py ${fs_util_file} ${LOCAL_OUTPUT}/${RESULT} ${PANGU_kHighConsume}"
    python2.7 ${PROJECT_PATH}/translator/chain_merge_and_sample.py ${fs_util_file} ${LOCAL_OUTPUT}/${RESULT} ${PANGU_kHighConsume}


#    NOTICE "cp ${LOCAL_OUTPUT}/${RESULT} -> pangu: ${PANGU_TMP}"
#    fs_util cp "${LOCAL_OUTPUT}/${RESULT}" ${PANGU_TMP}/${RESULT}
#
#    NOTICE "mv ${PANGU_TMP}/${RESULT} -> ${PANGU_kHighConsume}/${RESULT}"
#    fs_util mv ${PANGU_TMP}/${RESULT} ${PANGU_kHighConsume}/${RESULT}
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


