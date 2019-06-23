tfcluster=in-tensorflow
pn=2
wn=12
wc=2
pm=8192
wm=4096
runner="/usr/bin/tf_tool run -e lingxuan.mx/117776 -c ${tfcluster} -u admin  -p ./  -pn ${pn} -wn ${wn} -wc ${wc} -pm ${pm} -wm ${wm}"

hadoop fs -rmr hdfs://sf-iflow/user/admin/maxuan/dssm/v11_hdfs/checkpoint

$runner dssm.py --run_function train

#/usr/bin/tf_tool run dssm.py --run_function train
