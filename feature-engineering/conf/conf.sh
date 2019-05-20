#!/bin/bash
ONLINE="Y" # "Y" or "N"
ONLINE="N" # "Y" or "N"
alias rm="/bin/rm"

#=================重要，需要根据机房设置, la：洛杉矶机房，in：印度机房=========================

if [ "${ENV}X" = "X" ]; then
    ENV="in"
fi


if [ $ENV = "xx" ]; then
    ENV_NAME="xx"
    DATABASE="xx"
    HDFS_DATAMINING="hdfs://xx"

elif [ $ENV = "xx" ]; then
    ENV_NAME="xx"
    DATABASE="xx"
    HDFS_DATAMINING="hdfs://xx"
fi


if [ ${ONLINE} = "Y" ]; then
    EMAIL_ADDRESSEE="xx@xx.com xx@xx.com"
    year=$(date -d "-1 days" "+%Y")
    month=$(date -d "-1 days" "+%m")
    last_day=$(date -d "-1 days" "+%d")
elif [ ${ONLINE} = "N" ]; then
    EMAIL_ADDRESSEE="xx@xx.com"

fi

echo "ENV_NAME : ${ENV_NAME}"

hive="hive --hiveconf mapreduce.job.queuename=normal"
HIVE="hive --hiveconf mapreduce.job.queuename=normal"

HDFS_IFLOW="/xx"
HDFS_PROJECT_ROOT="xx/feature-engineering"
HDFS_TUIXING="xx"
HDFS_PYTHON3_PKG="${HDFS_DATAMINING}/xx/env/anaconda3.tar"
HDFS_MR_PKG="${HDFS_TUIXING}/MR_PKG"


# 定义字典
#declare -A PanguDirDict
# xx=([xx]="${xx}/xx" ["xx"]="${xx}/xx")
#
##打印指定key的value
#echo ${xx[xx]}