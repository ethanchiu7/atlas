#!/bin/bash
# author by Ethan Jo
# the main process should source common.sh

function HCAT_TOP_N() {
    # Usage : hdfs_file_path=$(GET_TOP_N abc/abc 2019 1)
    if [ ! $# -eq 4 ]; then
        FATAL "Usage: HCAT_TOP_N options"
        exit 1
    fi
    parent_directory=$1
    key_word=$2
    top_n=$3
    local_path=$4

    result_paths=$(hdfs dfs -ls ${parent_directory} | awk '{print $8}' | grep "${key_word}" | head -n ${top_n})
    STRING_EMPTY ${result_paths} "${parent_directory}/*${key_word}*"
    VERIFY_STATUS $?

    ERASE_FILE ${local_path}
    for i in ${result_paths}
    do
        NOTICE "save ${i}/part*  --> ${local_path}"
        hdfs dfs -cat ${i}/part* >> ${local_path}
    done
}

function HPUT() {
    if [ ! $# -eq 2 ]; then
        FATAL "Usage: HPUT local_file hdfs_path"
        exit 1
    fi
    if [ ! -f "$1" ]; then
        FATAL "local_file [$1] not exist"
        exit 1
    fi
    hdfs dfs -test -d "$2"
    if [ ! $? -eq 0 ]; then
        hdfs dfs -mkdir -p "$2"
    fi
    hdfs dfs -put -f "$1" "$2"
}

function HGET() {
    if [ ! $# -eq 2 ]; then
        FATAL "Usage: HGET hdfs_file local_file"
        exit 1
    fi

    hdfs dfs -test -f $1
    if [ ! $? -eq 0 ]; then
        FATAL "hdfs_file [$1] not exist"
        exit 1
    fi

    if [ -f $2 ]; then
        rm -f $2
    fi

    local_path=$(dirname $2)
    if [ ! -d ${local_path} ]; then
        mkdir -p ${local_path}
    fi

    hdfs dfs -get $1 $2
}

function HRM() {
    if [ ! $# -eq 1 ]; then
        FATAL "Usage: HRM hdfs_file_path"
        exit 1
    fi
    hdfs dfs -test -e $1
    if [ ! $? -eq 0 ]; then
        NOTICE "HDFS_PATH [$1] not exist"
        return 0
    fi
    hdfs dfs -rm -r -f $1
}

function ARCHIVE_FILES_TO_HDFS() {
    if [ $# -le 2 ]; then
        FATAL "Usage: ARCHIVE_FILES_TO_HDFS files_arr package_name hdfs_path"
        exit 1
    fi

    name=$1[@]
    job_name=$1
    hdfs_path=$2

    files_arr=("$@")
    ((last1_id=${#files_arr[@]} - 1))
    ((last2_id=${#files_arr[@]} - 2))
    job_name=${files_arr[last2_id]}
    hdfs_path=${files_arr[last1_id]}
    unset files_arr[last1_id]
    unset files_arr[last2_id]

    ERASE_DIR ${job_name}
    cd ${job_name}

    for i in "${files_arr[@]}"
    do
       echo "cp ${i} -> $(pwd;)"
       cp ${i} ./
    done
    tar -zcvf ./${job_name}.tar ./*

    NOTICE "$(pwd;)/${job_name}.tar --> ${hdfs_path}"
    HPUT ./${job_name}.tar ${hdfs_path}

    NOTICE "cd .. && rm -rf ./${job_name}"
    cd .. && rm -rf ./${job_name}

}