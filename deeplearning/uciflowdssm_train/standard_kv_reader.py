import threading
import logging
import traceback
import time
import tensorflow as tf
import math
import tf_util.kafka.kafka_reader as kafka_reader
#import tf_util.hdfs.hdfs_reader as hdfs_reader
import hdfs_reader
import random

def parseRecord(record, isTrain=True):
    raw_line_list = record.strip().split("\n")
    item_type = ""
    for kv in raw_line_list:
        if kv.startswith("item_type"):
            item_type = kv[kv.index(":")+1:]
    if(isTrain == True and (item_type != "0" and item_type != "201" and item_type == "208")):
        return None
    if(isTrain == False and (item_type != "0" and item_type != "201")):
        return None
    return record


class StandardKvKafkaReader(threading.Thread):
    def __init__(self,
            bootstrap_servers,
            group_id,
            max_partition_fetch_bytes,
            topic,
            batch_size,
            worker_index,
            total_workers,
            auto_offset_reset,
            metric_interval = 60,
            auto_metric=True,
            seek_to_end=False,
            qsize=2000,
            is_train=False):
        super(StandardKvKafkaReader, self).__init__()

        self.logger = logging.getLogger(__name__)
        self._batch_size = batch_size
        self._stop = False
        self._sess = None
        self._queue = None
        self._qsize = qsize
        self.is_init = False
        self._total_workers = total_workers
        self._is_train = is_train
        print('total worker is {}, index is {}'.format(total_workers, worker_index))
        print('bootstrap_servers is {}, topic is {}, group_id is {}, max_partition_fetch_bytes is {}'.format(bootstrap_servers, topic, group_id, max_partition_fetch_bytes))
        self.reader = kafka_reader.KafkaReader(topic=topic,
                             bootstrap_servers=bootstrap_servers,
                             group_id=group_id,
                             consumer_num=total_workers,
                             auto_offset_reset=auto_offset_reset,
                             auto_metric=auto_metric,
                             seek_to_end=seek_to_end,
                             consumer_index = worker_index,
                             max_partition_fetch_bytes=max_partition_fetch_bytes)

        self.q_raw_input = None
        self.__enqueue_op = None
        self.raw_input = None

    def init(self):
        if not self.is_init:
            self._queue = tf.FIFOQueue(self._qsize, tf.string)
            #self._queue = tf.RandomShuffleQueue(capacity=self._qsize,min_after_dequeue=1, dtypes=tf.string)
            self.q_raw_input = tf.placeholder(tf.string, name='raw_input')
            self.__enqueue_op = self._queue.enqueue(self.q_raw_input)
            self.raw_input = self._queue.dequeue()
            self.is_init = True

    def set_session(self, sess):
        self._sess = sess

    def get_raw_input_tensor(self):
        return self.raw_input

    def _do_enqueue_kafka(self, data):
        feed_dict = {self.q_raw_input: data}
        self._sess.run(self.__enqueue_op, feed_dict=feed_dict)

    def run(self):
        if self._sess is None:
            self.logger.error('Session is None, set session first.')
            return
        self.logger.info('Start preprocess data thread.')
        steps = 1
        data = []
        while not self._stop:
            while (len(data) < self._batch_size):
                receive = self.reader.get_messages(self._batch_size, timeout=1000)
                length = len(receive)
                for i in range(length):
                    raw_line = receive[i]
                    record = raw_line[1]
                    record = parseRecord(record, self._is_train)
                    if(record == None):
                        continue
                    data.append(record)
            # fucntion = train
            self._do_enqueue_kafka(data[:self._batch_size])
            data = data[self._batch_size:]

            #if iter % 100 == 0:
            #    print("[{0}] read kafka {1} records.".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), count_sum))
            #    # print("read extra kafka %d records." % extra_count_sum)

    def stop(self):
        self._stop = True

class StandardKvHdfsReader(threading.Thread):
    def __init__(self,
            data_paths,
            batch_size,
            record_flag="[add]",
            cycle=1,
            shuffle=False,
            total_num_worker=1,
            worker_index=0,
            qsize=100,
            reader_checkpoint=None,
            begin_time=None,
            dir_depth=2):

        '''
        data_path    :    source path,Comma separated
        clycle       :    reader all the file cycle time
        shuffle      :    shuffle to read data
        total_num_worker : total worker to cousume
        worker_index :     worker index
        '''

        super(StandardKvHdfsReader, self).__init__()
        self._sess = None
        self._reader = hdfs_reader.HdfsReader(data_paths, cycle, shuffle, total_num_worker, worker_index, reader_checkpoint=reader_checkpoint
                                              #reader_checkpoint=reader_checkpoint, begin_time=begin_time, dir_depth=dir_depth)
                                              )
        self._batch_size = batch_size
        self._record_flag = str(record_flag)
        # record_flag must be string, unicode will lead to string.find() very slowly
        self._record_separate = '\n' + self._record_flag + "\n"

        self._queue = None
        self.q_raw_input = None
        self.__enqueue_op = None

        self.raw_input = None
        self._stop = False
        self._qsize = qsize
        self.is_init = False
        self.total_num_worker = total_num_worker
        self.worker_index = worker_index
        self.worker_status_queue = None
        self.__worker_status_queue_enqueue_op = None
        self.__worker_status_queue_size = None
        self._reader_buf = None
        self._buf_begin_index = 0
        self._count = 10000

    def init(self):
        if not self.is_init:
            self._queue = tf.FIFOQueue(self._qsize, tf.string)
            self.q_raw_input = tf.placeholder(tf.string, name='raw_input')
            self.__enqueue_op = self._queue.enqueue(self.q_raw_input)
            self.raw_input = self._queue.dequeue()
            self._data_queue_size =  self._queue.size()
            if self.total_num_worker > 1:
                with tf.device('/job:ps/task:0'):
                    self.worker_status_queue = tf.FIFOQueue(self.total_num_worker, tf.int32, shared_name="wd_worker_status_shared_queue")
                    self.__worker_status_queue_enqueue_op = self.worker_status_queue.enqueue(self.worker_index)
                    self.__worker_status_queue_size = self.worker_status_queue.size()
            self.is_init = True

    def set_session(self, sess):
        self._sess = sess

    def get_one_sample(self):
        if not self._reader_buf:
            self._reader_buf = '\n'.join(self._reader.get_messages(count = self._count))
        flag_index = self._reader_buf.find(self._record_separate, self._buf_begin_index)
        if flag_index == -1:
            old_buf_reserve = self._reader_buf[self._buf_begin_index:]
            new_buf = '\n'.join(self._reader.get_messages(count = self._count))
            self._reader_buf = old_buf_reserve + '\n' + new_buf
            self._buf_begin_index = 0
            return self.get_one_sample()
        record = self._reader_buf[self._buf_begin_index:flag_index]
        self._buf_begin_index = flag_index + 1
        return record

    def get_batch_data(self, batch_size):
        batch_data = []
        for i in range(batch_size):
            sample = self.get_one_sample()
            batch_data.append(sample)
        return batch_data

    def get_raw_input_tensor(self):
        return self.raw_input

    def save_and_check_worker_statues(self):
        print("worker %d finished reader " % self.worker_index)
        if self.total_num_worker <= 1:
            return True
        self._sess.run(self.__worker_status_queue_enqueue_op)
        queue_size = self._sess.run( self.__worker_status_queue_size)
        print("qsize: %d , total woker number: %d" % (queue_size, self.total_num_worker))
        if queue_size >= self.total_num_worker:
            return True
        else:
            return False

    def check_data_queue_empty(self):
        data_queue_size = self._sess.run(self._data_queue_size)
        if(data_queue_size == 0):
            return True
        return False

    def run(self):
        count = 0
        thread = threading.current_thread()
        thread_name = thread.getName()
        data = []
        while not self._stop:
            try:
                while (len(data) < self._batch_size):
                    receive = self.get_batch_data(self._batch_size)
                    length = len(receive)
                    for i in range(length):
                        record = receive[i]
                        data.append(record)
                feed_dict={self.q_raw_input: data[:self._batch_size]}
                data = data[self._batch_size:]
                self._sess.run(self.__enqueue_op, feed_dict=feed_dict)
                if count < 10000 and count % 1000 == 0:
                    print ("************* %s read %d batch, %d data! **************"
                           % (thread_name, count, count * self._batch_size))
                count += 1
            except IOError as e:
                self.stop()
                if self.save_and_check_worker_statues() and self.check_data_queue_empty():
                    self._sess.close()
            except RuntimeError as e:
                self.stop()
        print ("************* Finally: %s read %d batch, %d data! **************"
               % (thread_name, count, count * self._batch_size))

    def stop(self):
        self._stop = True
