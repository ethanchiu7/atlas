import codecs
import numpy
import sys
import os
import pandas
import time

local_root = '/home/admin/maxuan/runspace/iflow_match/deep_match/deep_match_base2/'
local_file_root = local_root + 'data/'

def GetMergeHdpFile(hdp_dir , local_path):
    code_return = os.system('hadoop fs -getmerge %s %s '%(hdp_dir , local_path))
    return code_return

if sys.argv[1] == 'r1':
    hadoop_file_path = 'hdfs://in-iflow/user/iflow/maxuan/dssm/deep_match_top_items'
    local_file_path = local_file_root + 'deep_match_top_items.txt'
    if(os.path.exists(local_file_path)):
        os.remove(local_file_path)
    GetMergeHdpFile(hadoop_file_path, local_file_path)


if sys.argv[1] == 'r2':
    local_file_path = local_file_root + 'deep_match_top_items.txt'
    names=['item_id', 'lang', 'publish_tm', 'cate_id', 'count']
    table = pandas.read_table(local_file_path, names=names)
#    publish_tm_filter = (int(time.time()) - 3600 * 24 * 5) * 1000
#    table.drop(table[table['publish_tm']<publish_tm_filter].index, inplace=True)
    language_list = table['lang'].unique()
    for lang in language_list:
        table_select = table[table['lang'] == lang]
        local_file_lang_path = local_file_root + lang
        table_select.to_csv(local_file_lang_path, sep = ' ', header=False, index=False)
    os.system('cd %s;rm -f data.tar;tar cvf data.tar *'%(local_file_root))
