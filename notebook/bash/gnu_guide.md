# awk

# sed

    hive -f ${SQL_FILE} | sed 's/\t/,/g' >> ${HIVE_RESULT}

# tr

    hive -e "select * from pms.pms_tp_config" | tr "\t" ","