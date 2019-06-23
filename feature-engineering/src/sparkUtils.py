import os
from pyspark import SparkConf, SparkContext, StorageLevel


# def load_data_to_rdd(sc: SparkContext, data_path: str, seq='\t'):
#     rdd = sc.textFile(data_path) \
#         .map(lambda x: x.split(seq, 1)) \
#         .filter(lambda x: x and len(x) == 2 and x[0] and x[0] != "") \
#         .persist(StorageLevel.)

def load_local_file_to_dict_bc(sc, path, seq='\t', key_index=0, value_index=1):
    if str(path).startswith('/') and not str(path).startswith('file://'):
        path = 'file://' + path
    rdd = sc.textFile(path) \
            .map(lambda x: x.split(seq)) \
            .map(lambda x: (x[key_index], x[value_index])) \
            .filter(lambda x: x and len(x) >= 2 and x[0] and x[0] != "") \
            .persist(storageLevel=StorageLevel.MEMORY_ONLY)

    key_list = rdd.keys().collect()
    value_list = rdd.values().collect()
    kv_dict = dict(zip(key_list, value_list))
    kv_dict_bc = sc.broadcast(kv_dict)

    rdd.unpersist()
    return kv_dict_bc


def load_hdfs_file_to_dict_bc(sc, path, seq='\t', key_index=0, value_index=1):
    if str(path).startswith('/') and not str(path).startswith('hdfs://'):
        path = 'hdfs://' + path
    rdd = sc.textFile(path) \
            .map(lambda x: x.split(seq)) \
            .map(lambda x: (x[key_index], x[value_index])) \
            .filter(lambda x: x and len(x) >= 2 and x[0] and x[0] != "") \
            .persist(storageLevel=StorageLevel.MEMORY_ONLY)

    key_list = rdd.keys().collect()
    value_list = rdd.values().collect()
    kv_dict = dict(zip(key_list, value_list))
    kv_dict_bc = sc.broadcast(kv_dict)

    rdd.unpersist()
    return kv_dict_bc


def load_hdfs_file_rdd(sc, path, level=StorageLevel.MEMORY_AND_DISK):
    # if not str(path).startswith('hdfs://'):
    #     path = 'hdfs://' + path
    rdd = sc.textFile(path) \
            .map(lambda x: x.split(seq, 1)) \
            .filter(lambda x: x and len(x) == 2 and x[0] and x[0] != "") \
            .persist(storageLevel=level)
    return rdd


def save_rdd_hdfs(rdd, path):
    # if not str(path).startswith('hdfs://'):
    #     path = 'hdfs://' + path

    path_tmp = path.strip('\\/') + ".tmp"
    cmd = 'hadoop fs -rm -r ' + path_tmp
    os.system(cmd)
    rdd.saveAsTextFile(path_tmp)
    cmd = 'hadoop fs -rm -r %s; hadoop fs -mv %s %s'%(path, path_tmp, path)
    os.system(cmd)
    # NOTICE("save rdd to %s success"%path)


COMMON_SC = [("spark.executor.memory", '5g'),
             ("spark.kryoserializer.buffer.max","512M"),
             ("spark.rpc.askTimeout", "600s"),
             ("spark.network.timeout", "1000s"),
             ("spark.storage.memoryFraction", "0.2"),   # Decrease the fraction of memory reserved for caching
             ("spark.driver.memory", "50g"),
             ("spark.memory.offHeap.enabled", "true"),
             ("spark.memory.offHeap.size", "30g")   # You can increase the offHeap size if you are still facing the OutofMemory issue
             ]


def get_SparkContext(app_name='tuixing-spark', **kwargs):
    conf = SparkConf()
    conf.setAppName(app_name)
    conf.setAll(COMMON_SC)
    for key in kwargs:
        conf.set(key, kwargs[key])

    sc = SparkContext(conf=conf)
    return sc

# def save_file_hdfs(file, path):
#     path_tmp = path.strip('\\/') + ".tmp"
#     cmd = 'hadoop fs -rm -r %s; hadoop fs -mkdir %s'%(path_tmp, path_tmp)
#     os.system(cmd)
#     cmd = 'hadoop fs -put %s %s'%(file, path_tmp)
#     os.system(cmd)
#     cmd = 'hadoop fs -rm -r %s; hadoop fs -mv %s %s'%(path, path_tmp, path)
#     os.system(cmd)
#     NOTICE("save file to %s success"%path)


# def ReadHdfsPaths(hdfs_root):
#     paths = set()
#     status, output = commands.getstatusoutput('hadoop fs -ls ' + hdfs_root)
#     if status != 0:
#         return paths
#     output = output.replace('\n', ' ')
#     items = output.split(' ')
#     for item in items:
#         if item.startswith('hdfs'):
#             paths.add(item)
#     return paths

# def GetMultiPaths(hdfs_root, day_num):
#     paths_new = []
#     for i in xrange(day_num):
#         time_bgn = (datetime.datetime.today() - datetime.timedelta(hours = 24 * i)).strftime("%Y-%m-%d")
#         path = hdfs_root.strip('/') + '/' + time_bgn
#         paths_new.append(path)
#     paths_new = set(paths_new)
#     paths_old = ReadHdfsPaths(hdfs_root)
#
#     paths = list(paths_new & paths_old)
#     NOTICE('path: {0}'.format(paths))
#     return paths
