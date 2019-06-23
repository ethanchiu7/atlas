#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf

def scan_file(path):
    file_list = []
    for f in tf.gfile.Walk(path):
        directory = f[0]
        for name in f[2]:
            file_list.append(os.path.join(directory, name))
    return file_list

def scan_dir(path, path_list):
    file_list = []
    for f in path_list:
        file_list.extend(scan_file(path + '/' + f))
    return sorted(list(set(file_list)))

def clean_expired_models(model_root, reserve_count=60):
    try:
        dirnames = tf.gfile.ListDirectory(model_root)
        digit_list = [d for d in dirnames if d.isdigit()]
        dirnames = sorted(digit_list, key = lambda x: int(x), reverse = True)
        need_remove = dirnames[reserve_count:]
        if not need_remove:
            print ('Total [%d] model in root[%s], less than threshold[%d]. No need to clean.'\
                            %(len(dirnames), model_root, reserve_count))
        else:
            print ('Total [%d] model in root[%s], more than threshold[%d]. Clean expired models.'\
                            %(len(dirnames), model_root, reserve_count))
        for dirname in need_remove:
            full_path = os.path.join(model_root, dirname)
            print ('Remove expired dir[%s]' % full_path)
            tf.gfile.DeleteRecursively(full_path)
    except tf.errors.OpError, e:
        msg = 'Remove expired model from [%s] failed. reason: %s' %(model_root, e)
        print ('clean_expired_models error, error msg = [%s]' % msg)