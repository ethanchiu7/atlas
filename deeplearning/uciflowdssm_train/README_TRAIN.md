
参数配置
启动信息: tensorflow
参数名	参数说明	值
tf_script	运行脚本	dssm.py --run_function train
dependent_files	依赖文件	custom_metric.py,init_embeddings_from_checkpoint_helper.py,standard_kv_reader.py,dssm_input_schem...
work_dir	运行目录	/home/admin/maxuan/runspace/iflow_match/deep_match/deep_match_base2
docker_image	docker镜像	docker-hub.ucweb.com:5000/shenma/tensorflow_r1.4:opt_2019_03
amonitor_service_name	Amonitor服务名称	无
status_node_id	状态服务id	无
algorithm_name	算法名称	无
algorithm_version	算法版本号	无
algorithm_args	算法运行的参数配置	{}
web_ide_distribute_mode	以WEB IDE分布式模式运行	false
project_id	WEB IDE项目ID	无
version	WEB IDE项目版本	无
classification	运行分类	plugin
运行方式: 分布式运行
参数名	参数说明	值
ps_num	ps实例数	4
ps_cpu	单个ps实例占用的cpu core数	1
ps_gpu	单个ps实例占用的gpu卡数量	0
ps_memory	ps内存	10240M
worker_num	worker实例数	14
worker_cpu	单个worker实例占用的cpu core数	2
worker_gpu	单个worker实例占用的gpu卡数量	0
worker_memory	worker内存	10240M
chief_worker_memory_boost	worker 0内存膨胀系数	4
resource_adjust_config	动态资源调整策略	not_use_resource_adjust
yarn_worker_failed_retry_count	container最大尝试运行次数（TF任务需配1）	1
container_allocate_timeout_s	container等待分配超时秒数	180
node_labels	node labels	无
start_container_when_allocated	每个container分配成功后立即启动	false
wait_all_worker_term	等待所有worker结束再退出	false
reserve_exited_docker_container	任务结束后，保留docker container	false
need_profile_tf_process	打开tensorflow性能分析	true
hadoop集群相关配置: cluster_config
参数名	参数说明	值
queue	任务队列	无
hadoop_conf	hadoop配置	<USE_HADOOP_CONF_IN_IMAGE>
zookeeper	zookeeper地址	无
输出: tensorflow
参数名	参数说明	值
log_dir	模型输出路径	hdfs://in-tensorflow/user/admin/maxuan/dssm/v11_hdfs/checkpoint
任务基本信息
任务id:160659任务名:train 启动机器列表:11.6.116.147上次运行机器:11.6.116.147执行用户:admin超时报警:无超时自杀:无
任务宏( 什么是任务宏?)
宏名	值	上次运行时值
__TASK_ID__		160659
__TASK_NAME__		train
__USER__		admin