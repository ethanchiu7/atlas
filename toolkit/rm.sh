#!/bin/bash

# add the followng script into ~/.bashrc
# alias rm=PATH_TO_THIS_SCRIPT/rm.sh
# alias rmdir=PATH_TO_THIS_SCRIPT/rm.sh

trash_root=/home/zhaoxin/trash

for src_full_path in $@; do
    if [ "${src_full_path}" == "" ]; then echo "fail!"; fi

    if [ "${src_full_path:0:1}" == "-" ]; then
        log="Skip path: [${src_full_path}]"
        echo ${log} >> ${trash_root}/log
        echo ${log}
        continue
    fi

    if [ ! "${src_full_path:0:1}" == "/" ]; then
        src_full_path=$(pwd)/${src_full_path}
    fi
    src=$(echo ${src_full_path} | awk -F '/' '{if ($NF=="") print $(NF-1); else print $NF}')
    trash=${trash_root}/$(date +%Y%m%d)
    if [ ! -d ${trash} ]; then mkdir ${trash}; fi
    dst=${trash}/${src}.$(date +%H%M%S)
    mv ${src_full_path} ${dst}
    log="Move [${src_full_path}] to [${dst}]"
    echo ${log} >> ${trash_root}/log
    echo ${log}
done

exit 0