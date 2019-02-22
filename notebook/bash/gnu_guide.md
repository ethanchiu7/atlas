# Lists of Commands

    command1 && command2
command2 is executed if, and only if, command1 returns an exit status of zero (success)

    command1 || command2
command2 is executed if, and only if, command1 returns a non-zero exit status.

The return status of AND and OR lists is the exit status of the last command executed in the list

    command1 ; command2

# awk

# sed

    hive -f ${SQL_FILE} | sed 's/\t/,/g' >> ${HIVE_RESULT}

# tr

    hive -e "select * from pms.pms_tp_config" | tr "\t" ","

# readlink for Mac

    MacPorts and Homebrew provide a coreutils package containing greadlink (GNU readlink). Credit to Michael Kallweitt post in mackb.com.

    brew install coreutils

    greadlink -f file.txt

    alias readlink=greadlink