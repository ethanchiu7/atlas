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

local_xgboost_model="${PROJECT_PATH}/data/iflow_general_gbm_model_xgb_00.model"
local_xgboost_dump="${PROJECT_PATH}/data/iflow_general_gbm_model_xgb_00.dump"

function split_train_test() {
    echo "split_train_test ..."
    cat ${PROJECT_PATH}/data/train_sample_20190507_200731.smv | \
        python ${PROJECT_PATH}/src/split_train_test.py 5 test > ${PROJECT_PATH}/data/train_sample_20190507_200731.smv.test
    cat ${PROJECT_PATH}/data/train_sample_20190507_200731.smv | \
        python ${PROJECT_PATH}/src/split_train_test.py 5 train > ${PROJECT_PATH}/data/train_sample_20190507_200731.smv.train
}

function train_xgb() {
    echo "train_xgb ..."
    xgboost ${PROJECT_PATH}/conf/xgboost.conf \
    task=train \
    data=${PROJECT_PATH}/data/train_sample_20190507_200731.smv.train \
    eval[valid]=${PROJECT_PATH}/data/train_sample_20190507_200731.smv.test \
    model_out=${local_xgboost_model}

#    xgboost ${PROJECT_PATH}/conf/xgboost.conf \
#    task=dump \
#    fmap=conf/featmap.txt \
#    model_in=${local_xgboost_model} \
#    name_dump=${local_xgboost_dump}


}

function update_model() {
    echo "update_model ..."
    rm -f
    bzip2 -f ${local_xgboost_model}
    md5sum ${local_xgboost_model}.bz2 | awk '{print $1}' > ${local_xgboost_model}.md5

    #todo mv
}


function compare_baseline() {
    echo "compare to prior CTR"
}

split_train_test

train_xgb

update_model
