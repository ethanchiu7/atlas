#! /bin/env python
#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Queue
import time
import sys
from random import shuffle as Shuffle
import logging

class HdfsReader():
    def __init__(self, data_paths, cycle=1, shuffle=False, total_num_worker=1, worker_index=0, is_contain_folder=False,reader_checkpoint=None):
        '''
        data_path          :    source path,Comma separated
        clycle             :    reader all the file cycle time
        shuffle            :    shuffle to read data, unit is file
        total_num_worker   :    total worker of tensorflow to consume
        worker_index       :    worker index of tensorflow
        is_contain_folder  :    if contains folder in data_path or not
        '''
        self._cycle = cycle
        self._data_paths = data_paths.split(",")
        self._file_queue = Queue.Queue()
        self._total_num_worker= total_num_worker
        self._worker_index = worker_index
        self._reader_checkpoint = reader_checkpoint
        self._tmp_file_name = set(["_temporary", "_SUCCESS"])
        self._run_time = time.time()
        self._checkpoint_files = self.scan_file(reader_checkpoint)
        self._history_processed_files = self.read_sets_from_files(self._checkpoint_files)
        print("history_processed_files")
        print(self._history_processed_files)
        self._last_read_file = None
        self._logger = logging.getLogger(__name__)
        self._logger.addHandler(logging.StreamHandler())
        self._logger.setLevel(logging.INFO)
        self._logger.info('data_path is %s,cycle is %s,shuffle is %s,total_num_worker is %s,worker_index is %s' % (data_paths, cycle, shuffle, total_num_worker, worker_index)) 
        tmp_set = set()
        for data_path in self._data_paths:
            files = set(self.scan_file(data_path))
            tmp_set = tmp_set | files
        self._all_files = sorted(list(tmp_set))
        for tmp in self._all_files:
            if tmp not in self._history_processed_files:
                self._file_queue.put(tmp)

        tmp_dir_list = []
        total_file = 0
        while (self._file_queue.qsize() > 0):
            tmp_file = self._file_queue.get()
            if(total_file % self._total_num_worker == self._worker_index):
                tmp_dir_list.append(tmp_file)
            total_file = total_file + 1

        if shuffle:
           Shuffle(tmp_dir_list)
        print("train file list")
        print(tmp_dir_list)
        for file_name in tmp_dir_list:
            self._file_queue.put(file_name)

        self._file_handle = None
        self._length = self._file_queue.qsize()
        self._logger.info("qsize is %s" %(self._length))
        self._file_consumed_number = 0
        self._total_file_to_consumed = self._length * cycle


    def get_current_cycle(self):
        return int(self._file_consumed_number/self._length)

    def _get_message(self):
        if self._file_handle is None:
            if self._file_queue.empty():
                raise IOError('no messages')
            filename = self._file_queue.get()
            self._current_read_file = filename
            print("current_read_file: %s"%filename)
            self._file_handle = tf.gfile.GFile(filename, mode="r")
            self._file_queue.put(filename)
        value = self._file_handle.readline()
        if not value:
            self._file_handle.close()
            self._file_handle = None
            self._file_consumed_number = self._file_consumed_number + 1
            self._last_read_file = self._current_read_file
            print("last_read_file: %s"%self._last_read_file)
            self.save_checkpoint()
            return self._get_message()
        value = value.strip()
        return value

    def get_messages(self, count=1):
        if (self._total_file_to_consumed <= self._file_consumed_number):
            raise IOError('no messages')
        records = []
        for i in xrange(count):
            if (self._total_file_to_consumed <= self._file_consumed_number):
                raise IOError('no messages')
            try:
                records.append(self._get_message())
            except Exception as e:
                print(e)
                raise IOError('no messages')

        return records

    def _get_file_name_from_dir(self, data_path, queue):
        for filename in tf.gfile.ListDirectory(data_path):
            path_tmp = data_path + "/" + filename
            if(tf.gfile.IsDirectory(path_tmp)):
                self._get_file_name_from_dir(path_tmp, queue)
            else:
                queue.put(path_tmp)

    def scan_file(self, path):
        file_list = []
        if not path:
            return file_list
        if not tf.gfile.Exists(path):
            return file_list
        if tf.gfile.IsDirectory(path):
            for filename in tf.gfile.ListDirectory(path):
                if filename in self._tmp_file_name:
                    continue
                path_tmp = path + "/" + filename
                if tf.gfile.IsDirectory(path_tmp):
                    file_list.extend(self.scan_file(path_tmp))
                else:
                    file_list.append(path_tmp)
        return file_list

    def read_sets_from_files(self, file_list):
        result = []
        for f in file_list:
            file_handle = tf.gfile.GFile(f, mode="r")
            line = file_handle.readline()
            while line:
                result.append(line.strip())
                line = file_handle.readline()
        return set(result)

    def save_checkpoint(self):
        if not self._reader_checkpoint:
            return
        reader_checkpoint_worker = self._reader_checkpoint + "/" + str(self._worker_index)  + "-" + str(self._run_time)
        if len(self._last_read_file) > 0:
            if tf.gfile.Exists(reader_checkpoint_worker):
                file_handler = tf.gfile.Open(reader_checkpoint_worker, 'a+')
            else:
                file_handler = tf.gfile.Open(reader_checkpoint_worker, 'w+')
            file_handler.write(self._last_read_file)
            file_handler.write("\n")
            file_handler.close()

    def close(self):
        if (self._file_handle is not None):
            self._file_handle.close()
            self.save_checkpoint()

if __name__ == "__main__":
    reader = HdfsReader(data_paths="hdfs://sm-spark-new/user/ww110750/query_title_7b_1.pairwise.sample-3b", cycle=1, shuffle=False, is_contain_folder=False)
    begin_time = time.time()
    print(reader.get_messages(1))
    print(reader.get_current_cycle())
    for i in range(1000):
        messages = reader.get_messages(1000)
        use_time = time.time() - begin_time
        #print("qps is %d" %(i*1000/use_time))
