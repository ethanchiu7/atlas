cur_minute=`date +%M`
if [ $cur_minute -gt 60 ]; then
    echo "exit process"
    exit 0
fi
/usr/bin/python2.7 update_top_item.py r1;
/usr/bin/python2.7 update_top_item.py r2;
