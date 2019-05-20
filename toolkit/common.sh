#!/bin/bash
# author by zhaoxin

function NOTICE()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo '[NOTICE] '$time' FUNCNAME:['${FUNCNAME[@]:1:7}'] '$1
}

function WARN()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo '[WARN] '$time' FUNCNAME:['${FUNCNAME[@]:1:7}'] '$1
}

function FATAL()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo '[FATAL] '$time' FUNCNAME:['${FUNCNAME[@]:1:7}'] '$1
}

function VERIFY_STATUS()
{

    # verify exit status of the last executed command
    if [ $# -lt 1 ]; then
        FATAL "the exit status of the last executed command should be given, EXIT 1"
        exit 1
    fi
    if [ $# -eq 1 ]; then
        status=$1
        if [ ${status} -ne 0 ]; then
            FATAL "last executed command failed , EXIT 1"
            exit 1
        fi
        return 0

    elif [ $# -eq 2 ]; then
        status=$1
        job_name=$2
        if [ ${status} -ne 0 ]; then
            FATAL "${job_name} failed , EXIT 1"
            exit 1
        fi
        return 0
    fi

    FATAL "VERIFY_STATUS : param ERROR, EXIT 1"
    exit 1
}

function STRING_EMPTY() {
    # empty return 1 else return 0
    if [ $# -eq 0 ];then
       echo "Usage: $0 string"
       exit 1
    fi
    msg=$2
    if [ "ZHAOXIN" = "ZHAOXIN${1}" ]; then
        NOTICE "${msg} that is EMPTY"
        return 1
    fi
    return 0
}

function ENSURE_DIR() {
    dir=$1
    if [ ! -d ${dir} ]; then
        mkdir -p ${dir}
    fi
}

function ENSURE_DIRS() {
    if [ $# -eq 0 ];then
       echo "Usage: $0 params"
       exit 1
    fi
    for arg in $@
    do
        ENSURE_DIR ${arg}
    done
}

function REMOVE() {
    if [ $# -ne 1 ]; then
        FATAL "Usage : $0 file_path"
        exit 1
    fi
    if [ ! -f $1 ]; then
        return 0
    fi
    rm -rf $1
    VERIFY_STATUS $? "REMOVE ${1}"
}

function ERASE_FILE() {
    if [ $# -ne 1 ]; then
        FATAL "Usage : $0 file_path"
        exit 1
    fi
    REMOVE $1
    touch $1
}

function ERASE_DIR() {
    dir=$1
    if [ ! -d ${dir} ]; then
        mkdir -p ${dir}
    else
        rm -rf ${dir}
        mkdir -p ${dir}
    fi
}
