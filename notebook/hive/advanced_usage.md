# memory set
    set mapreduce.map.memory.mb=4096;
    set mapreduce.map.java.opts=-Xmx5200m;
    set mapreduce.reduce.memory.mb=4096;
    set mapreduce.reduce.java.opts=-Xmx5200m;
    
    
# set variable into sql file
    D1=`date +"%Y-%m-%d"`
    D2=`date +"%Y-%m-%d" -d "1 day ago"`
    hive -hivevar date1=$D1 -hivevar date2=$D2 -f test.sql > test.rst
    
# set transform into hsql
    # test.sql ----
    
    add file transform.py;
    
    set mapreduce.map.memory.mb=4096;
    set mapreduce.reduce.memory.mb=4096;
    set mapreduce.job.queuename=root.normal;
    
    use dbname;
    
    select
    transform(column_1,column_2,column_3) using 'python transform.py' as (out_1,out_2)
    from tblname
    where
        length(column_xx) > 0 and
        (log_date = '${hivevar:date1}' or log_date = '${hivevar:date2}') 
    ;
    
# title_keyword.py
    # 类似 map.py 写法 
    for line in sys.stdin:
        (column_1,column_2,column_3) = line.split('\t')
        ...