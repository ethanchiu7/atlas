#encoding=utf-8
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import sys
import json
import time
import smnn
import file_op
from variable_reuse_helper import InitEmbeddingsHook
import traceback
from model_builder import WideNDeepModelBuilder
from smnn.io.parser.sm_kv_record_parser import SmKVRecordParser
from smnn.io.input import SmKVRecordKafkaInput, SmKVRecordFileInput
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.training import saver
from tensorflow.python.lib.io.file_io import stat as Stat
from tensorflow.python.status_service import status_tool

class Runner(object):
    def __init__(self):
        self.args = smnn.Args()
        self.context = smnn.Context()
        self.context.deserialize_from_args(self.args)
        self.job_name = self.context.get_job_name()
        self.task_index = self.context.get_task_index()
        self.cluster_conf = self.args.declare_and_get_arg('cluster_conf', str, 'cluster_conf.json', False)
        self.run_function = self.args.declare_and_get_arg('run_function', str, None, False)
        self.smnn_schema = self.load_config_file('smnn_schema', 'wd_smnn_schema.json')
        self.parse_model_desc()
        self.model_fn_param = {}

    def parse_model_desc(self):
        model_desc = self.load_config_file('model_desc', 'wd_model_desc.json')
        self.model_desc_obj = json.load(open(model_desc, "r"))
        self.model_parameters = self.model_desc_obj["modelConfig"]["parameters"]
        self.label = self.model_parameters["label"]
        self.record_flag = self.model_parameters["record_flag"]
        self.batch_size = self.model_parameters["batch_size"]
        self.save_checkpoints_secs = self.model_parameters["save_checkpoints_secs"]
        self.keep_checkpoint_max = self.model_parameters["keep_checkpoint_max"]
        self.init_dir = self.model_parameters.get("init_dir")
        self.export_path = os.path.join(self.model_parameters.get("path"), 'export')
        self.checkpoint_path = os.path.join(self.model_parameters.get("path"), 'checkpoint')
        self.model_parameters['checkpoint_path'] = self.checkpoint_path
        self.reader_checkpoint_path = os.path.join(self.model_parameters.get("path"), 'read_checkpoint')
        self.print_desc_info()

    def print_desc_info(self):
        print ('-' * 40)
        print("label: ", self.label)
        print("record_flag: ", self.record_flag)
        print("batch_size: ", self.batch_size)
        print("save_checkpoints_secs: ", self.save_checkpoints_secs)
        print("keep_checkpoint_max: ", self.keep_checkpoint_max)
        print("init_dir: ", self.init_dir)
        print("checkpoint_path: ", self.checkpoint_path)
        print("export_path: ", self.export_path)
        print("reader_checkpoint_path: ", self.reader_checkpoint_path)

    def load_config_file(self, name, default):
        config = self.args.declare_and_get_arg(name, str, default, False)
        if tf.gfile.Exists(config):
            return config
        config = 'config/' + config
        print (config, tf.gfile.Exists(config))
        if tf.gfile.Exists(config):
            return config
        return None

    def tensor_dict_from_kafka(self, parameters):
        parameters['need_ckpt'] = True
        if not 'train' in self.run_function:
            group_id = parameters.get("group_id")
            parameters['group_id'] = group_id + self.run_function
            parameters['seek_to_end'] = True
            parameters['need_ckpt'] = False
        bootstrap_servers = parameters['bootstrap_servers']
        group_id = parameters["group_id"]
        topic = parameters["topic"]
        
        input_uri = "kafka://{}?topic={}&record_flag={}".format(bootstrap_servers, topic, self.record_flag)
        tf.logging.info("Get data from kafka, uri[%s], group_id[%s]", input_uri, group_id)

        tensor_dict = SmKVRecordKafkaInput(
                input_uri,
                self.smnn_schema,
                batch_size=self.batch_size,
                group_id=group_id, 
                seek_to_begin=False,
                seek_to_end=True,
                max_partition_fetch_mb=4,
                need_checkpoint=parameters['need_ckpt'])
        return tensor_dict
    
    def tensor_dict_from_hdfs(self, parameters):
        cycle = parameters.get('cycle', 1)
        is_shuffle = parameters.get('is_shuffle', 0)
        mode = 'train' if 'train' in self.run_function else 'test'
        data_path = [os.path.join(parameters[mode]['data_path'], f)
                     for f in parameters[mode]['data_path_list']]
        data_path = ','.join(data_path)
        file_filter_pattern = ".*\.(crc|swp)"
        input_uri = '{}?record_flag={}'.format(data_path, self.record_flag)
        tf.logging.info("Get data from file, uri[%s]", input_uri)
        reader_checkpoint_path = self.reader_checkpoint_path
        if mode == 'test':
            reader_checkpoint_path = None
        tensor_dict = SmKVRecordFileInput(
                input_uri,
                self.smnn_schema,
                batch_size=self.batch_size,
                checkpoint_dir=reader_checkpoint_path,
                need_shuffle=is_shuffle,
                num_epochs=cycle,
                file_filter_pattern=file_filter_pattern)
        return tensor_dict

    def create_reader(self):
        source_type = self.model_desc_obj.get("dataSource").get("source_type")
        parameters = self.model_desc_obj.get("dataSource").get(source_type + "_parameters")
        print ('--------- create_reader parameters start ---------')
        for k in sorted(parameters.keys()):
            print (k, parameters[k])
        print ('---------- create_reader parameters end ----------')
        if source_type == "kafka":
            reader = self.tensor_dict_from_kafka(parameters)
        elif source_type == "file":
            reader = self.tensor_dict_from_hdfs(parameters)
        reader.init(self.context)
        return reader

    def input_fn_train(self):
        reader = self.create_reader()
        return reader.tensor_dict, tf.shape([-1])

    def serving_input_fn(self):
        kv_parser = SmKVRecordParser(self.smnn_schema, self.record_flag, '[common]')
        kv_parser.init()
        examples = tf.placeholder(dtype=tf.string, shape=[None])
        tensor_dict = kv_parser.get_tensor_dict(examples)
        receiver_tensors = {'examples': examples}
        return tf.estimator.export.ServingInputReceiver(tensor_dict, receiver_tensors)
    ######################## reader end ########################

    def gen_run_config(self):
        cluster = json.load(open(self.cluster_conf, "r"))
        cluster_spec = tf.train.ClusterSpec(cluster)
        tf_config = {
            'cluster': cluster,
            'task': {
                'type': self.job_name,
                'index': self.task_index
            }
        }
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        run_config = tf.contrib.learn.RunConfig(
                save_checkpoints_secs=self.save_checkpoints_secs,
                keep_checkpoint_max=self.keep_checkpoint_max)
        num_workers = len(cluster['worker'])
        num_ps = len(cluster['ps'])
        print ("**********************job base info **********************")
        print ("cluster:")
        print (cluster)
        print ("task type : %s" % self.job_name)
        print ("task index: %d" % self.task_index)
        print ("ps num : %d, work num : %d" % (num_ps, num_workers))
        print ("tf_confg: %s" % tf_config)
        return cluster_spec, run_config, num_workers

    def get_checkpoint_path(self, model_dir, sec_to_now):
        ckpt = saver.get_checkpoint_state(model_dir, None)
        if ckpt and len(ckpt.all_model_checkpoint_paths) > 0:
            path2tm = dict()  
            for i, p in enumerate(ckpt.all_model_checkpoint_paths):
                print ("^" * 40)
                st = Stat(p+".meta")
                tm = time.time() - st.mtime_nsec / 10**9
                path2tm[p] = tm
                print ("NO", str(i), ": ", p, " tm:", str(st.mtime_nsec / 10**9))
            sorted_paths = sorted(path2tm.items(), key=lambda x:x[1])
            print("sorted_paths: ", sorted_paths, "lenghts: ", len(sorted_paths))
            valid_paths = [x for x in sorted_paths if x[1] >= sec_to_now]
            print("valid_paths: ", valid_paths, "lengths: ", len(valid_paths))
            if len(valid_paths) > 0:
                return valid_paths[0]
            elif len(sorted_paths) > 0:
                return sorted_paths[0]
            return None
        return None            

    def train(self):
        if tf.gfile.Exists(self.cluster_conf):
            cluster_spec, run_config, num_workers = self.gen_run_config()
            server = tf.train.Server(cluster_spec,
                                     job_name=self.job_name,
                                     task_index=self.task_index)
        else:
            run_config = tf.contrib.learn.RunConfig(
                save_checkpoints_secs=self.save_checkpoints_secs)
            num_workers = 1
            server = None

        print ('-' * 40)
        print ('run_config =', run_config)

        if self.job_name == 'ps':
            print ("ps start")
            server.join()
        elif self.job_name == 'worker':
            print ("worker start")
            train_steps = self.model_parameters.get("train_steps")
            self.model_fn_param['run_config'] = run_config
            builder = WideNDeepModelBuilder(model_desc=self.model_desc_obj)
            model = builder.build_estimator(self.model_fn_param)
            hooks = []
            try:
                global_step = checkpoint_utils.load_variable(
                        self.checkpoint_path, tf.GraphKeys.GLOBAL_STEP)
            except:
                global_step = 0
            print ('global_step =', global_step)
            if self.init_dir and global_step < 100:
                print ('InitEmbeddings from %s' % self.init_dir)
                hooks.append(InitEmbeddingsHook(self.init_dir))
            model.train(input_fn=self.input_fn_train,
                        steps=train_steps,
                        hooks=hooks)

    def train_with_weight_column(self):
        self.model_fn_param['weight_column'] = 'weight'
        self.train()

    def evaluate(self):
        source_type = self.model_desc_obj.get("dataSource").get("source_type")
        if self.run_function == "evaluate_offline":
            source_type = "file"
        config = self.model_desc_obj.get("dataSource").get(source_type+'_parameters')
        test_steps = config['test_steps']
        builder = WideNDeepModelBuilder(model_desc=self.model_desc_obj)
        model = builder.build_estimator(self.model_fn_param)
        results = model.evaluate(input_fn=self.input_fn_train,
                                 steps=test_steps)
        status_tool.report_metric(results, results['global_step'])
        print ('=' * 40)
        for k in sorted(results.keys()):
            print (k, results[k])

    def evaluate_online(self):
        self.model_fn_param['metric_prefix'] = 'online_'
        self.evaluate()

    def evaluate_with_weight_column(self):
        self.model_fn_param['weight_column'] = 'read_tm_vl'
        self.model_fn_param['metric_prefix'] = 'weight_'
        self.evaluate()

    def evaluate_offline(self):
        self.model_fn_param['metric_prefix'] = 'offline_'
        self.evaluate()

    def evaluate_delay(self):
        source_type = self.model_desc_obj.get("dataSource").get("source_type")
        config = self.model_desc_obj.get("dataSource").get(source_type+'_parameters')
        ckpt_path_info = self.get_checkpoint_path(self.checkpoint_path, 7200)
        if not ckpt_path_info:
            print("Failed to get checkpoint from ", ckpt_dir)
            return
        print("evaluate checkpoint. path:%s, delay:%d sec." % (ckpt_path_info[0], int(ckpt_path_info[1])))
        test_steps = config['test_steps']
        builder = WideNDeepModelBuilder(model_desc=self.model_desc_obj)
        model = builder.build_estimator(self.model_fn_param)
        results = model.evaluate(input_fn=self.input_fn_train,
                                 steps=test_steps,
                                 checkpoint_path=ckpt_path_info[0])
        print ('=' * 40)
        for k in sorted(results.keys()):
            print (k, results[k])

    def export(self):
        export_dir = self.export_path
        builder = WideNDeepModelBuilder(model_desc=self.model_desc_obj)
        model = builder.build_estimator(self.model_fn_param)
        true_path = model.export_savedmodel(
                export_dir_base=export_dir, 
                serving_input_receiver_fn=self.serving_input_fn, 
                as_text=True)
        print ("export_dir: " + export_dir)
        print ("true_path: " + true_path)
        file_op.clean_expired_models(export_dir)

    def predict(self):
        source_type = self.model_desc_obj.get("dataSource").get("source_type")
        config = self.model_desc_obj.get("dataSource").get(source_type+'_parameters')
        builder = WideNDeepModelBuilder(model_desc=self.model_desc_obj)
        model = builder.build_estimator(self.model_fn_param)
        results = model.predict(input_fn=self.input_fn_train)
        print ('=' * 40)
        for i in range(self.batch_size):
            print ("score: ", results.next()['logistic'][0])

    def run(self):
        try:
            if self.run_function == "train":
                self.train()
            elif self.run_function == "train_with_weight_column":
                self.train_with_weight_column()
            elif self.run_function == "evaluate_online":
                self.evaluate_online()
            elif self.run_function == "evaluate_with_weight_column":
                self.evaluate_with_weight_column()
            elif self.run_function == "evaluate_offline":
                self.evaluate_offline()
            elif self.run_function == "evaluate_delay":
                self.evaluate_delay()
            elif self.run_function == "export":
                self.export()
            elif self.run_function == "predict":
                self.predict()
            else:
                print ("function [%s] not implement" %self.run_function)

        except tf.errors.CancelledError as e:
            print ("*" * 100)
            print ("wide and deep stop, because: %s" % e.message)
            print ("all wokers finished read data, program stoped !")
            time.sleep(60)
            print ("*" * 100)
            sys.exit(0)
        except Exception as e:
            print ("wide and deep stop, because: %s" % e.message)
            traceback.print_exc()
            sys.exit(1)
    
def main(_):
    print("CMD: {}".format(sys.argv))
    runner = Runner()
    runner.run()

if __name__ == "__main__":
    tf.app.run()
