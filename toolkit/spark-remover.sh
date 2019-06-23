#!/bin/bash
# tuixing.zx@alibaba-inc.com
#
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
    spark-remover
Options :
   chain_name
   -----------------------------------------------------------------------------
   chain_name:  |   translate_fun:     |  topN: |   hdfs_input:
   -----------  |   ---------------    |  ----  |  ---------------
   kCtrlItem    |   general_formate    |   -1   |  tuixing.zx/ctrl/seedname/result
   kWmoaCard    |   general_formate    |   -1   |  huirong.whr/wm_oa/wmoacard/result
   kWMOAEE      |   general_formate    |   -1   |  huirong.whr/wm_oa/wmoaee/indonesia/result
   kUser2Item   |   general_formate    |   -1   |  huirong.whr/iflow/iflowiib/result
   kHpGroupTag  |   general_formate    |   50   |  huirong.whr/homepage/gCateTag/indonesia/result
   kHpAttractive|   general_formate    |   -1   |  /user/datamining/yangmingmin/homepage/hctr_zipper_dev_hp/indonesian -> HP_ATTR
   kHpAttractive|   general_formate    |   -1   |  /user/datamining/yangmingmin/homepage/hctr_zipper_dev2_hp/indonesian -> HP_ATTR2
   kHpCate      |   general_formate    |   -1   |  /user/datamining/yangmingmin/homepage/hctr_zipper_dev/indonesian -> HP_CATE2 / HP_HIGH_CTR
   kHpCate      |   general_formate    |   -1   |  /user/datamining/yangmingmin/homepage/hctr_zipper_dev2/indonesian -> HP_CATE / HP_HIGH_CTR2



Example:
    ${FILE_NAME} kCtrlItem general_formate -1 tuixing.zx/ctrl/seedname/result
For more detail : http://lego-sf.sm.cn/#/flow/folder/121795/125091
EOT
}
if [ ! $# -eq 4 ]; then usage; exit -1; fi

HDFS_INPUT_FOLDER=${1}

RESULT_TIME=$(date +"%Y%m%d_%H%M%S")
RESULT_TS=$(date "+%s")
EXPIRE_DAYS=1


# ---- 注意不同集群环境变更此项 -----

NOTICE "DOC: "
CHAIN_NAME=${1}
TRANSLATE_FUN=${2}
TOPN=${3}
HDFS_INPUT=${4}
HDFS_OUTPUT="${HDFS_PROJECT_ROOT}/${CHAIN_NAME}/${RESULT_TIME}"
RESULT="${CHAIN_NAME}.multilang.${RESULT_TS}"

LOCAL_OUTPUT=${PROJECT_PATH}/data
ENSURE_DIR ${LOCAL_OUTPUT}

ENSURE_DIR ${PROJECT_PATH}/log

# Pangu Chain Dir
PANGU_DIR=$(eval echo '$'PANGU_${CHAIN_NAME})

function spark4translate() {

    NOTICE "CHAIN_NAME : ${CHAIN_NAME}"
    NOTICE "HDFS_INPUT : ${HDFS_INPUT}"
    NOTICE "HDFS_OUTPUT : ${HDFS_OUTPUT}"
    NOTICE "RESULT : ${RESULT}"

    NOTICE "RUN SPARK ..."
    export PYSPARK_PYTHON=/usr/local/bin/python2.7
    spark-submit \
    --master yarn-client \
    --queue normal \
    --num-executors 300 \
    --py-files ${PROJECT_PATH}/translator/sparkUtils.py \
    ${PROJECT_PATH}/translator/sparkTranslator.py ${HDFS_INPUT} ${HDFS_OUTPUT} ${TRANSLATE_FUN} ${TOPN}
    VERIFY_STATUS $? "run_spark_sparkTranslator"

    NOTICE "remove data to pangu ..."

    hdfs dfs -cat ${HDFS_OUTPUT}/* > ${LOCAL_OUTPUT}/${RESULT}

    NOTICE "CHAIN_NAME : ${CHAIN_NAME}"
    NOTICE "HDFS_INPUT : ${HDFS_INPUT}"
    NOTICE "HDFS_OUTPUT : ${HDFS_OUTPUT}"
    NOTICE "RESULT: ${LOCAL_OUTPUT}/${RESULT}"


}

function remove2pangu() {
    # move
    cd ${PROJECT_PATH}/log

    # for bug of pangu
    ENSURE_DIR "__chain_info_dir_DO_NOT_DELETE__"

    fs_util_file=$(which fs_util | head -n1 | awk -F '=' '{print $2}' | sed $'s/\'//g')

    NOTICE "run chain_merge_and_sample.py ..."
    NOTICE "This is for bug of pangu !!!"
    NOTICE "python2.7 ${PROJECT_PATH}/translator/chain_merge_and_sample.py ${fs_util_file} ${LOCAL_OUTPUT}/${RESULT} ${PANGU_DIR}"
    python2.7 ${PROJECT_PATH}/translator/chain_merge_and_sample.py ${fs_util_file} ${LOCAL_OUTPUT}/${RESULT} ${PANGU_DIR}

#    NOTICE "cp ${LOCAL_OUTPUT}/${RESULT} -> pangu: ${PANGU_TMP}/${RESULT}"
##    fs_util cp "${LOCAL_OUTPUT}/${RESULT}" "${PANGU_TMP}/${RESULT}"
##
##    # fs_util  mkdir pangu://SF-IFLOW/product/iflow_dynamic_data/zihan.cx_kSVideoContextB
##    NOTICE "mv ${PANGU_TMP}/${RESULT}-> ${PANGU_DIR}/${RESULT}"
##    fs_util mv ${PANGU_TMP}/${RESULT} ${PANGU_DIR}/${RESULT}
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

spark4translate

remove2pangu

clean4project

