#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os

import tensorflow as tf
import standard_kv_reader
from tensorflow.contrib.session_bundle import exporter
from model_builder import DssmBuilder
import tf_util.parser.sm_standard_kv_parser as lib_parser
import tf_util.parser.schema_parser as schema_parser
from shenma.common import json_util
from start_reader_hook import StartStandardKvReaderHook
from tf_util.ganglia import gmetric_writer
import traceback
import custom_metric
import time
import init_embeddings_from_checkpoint_helper as init_embeddings
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.training import saver
from tensorflow.python.lib.io.file_io import stat as Stat
from tensorflow.python.status_service import status_tool
from smnn.io.input import SmKVRecordKafkaInput, SmKVRecordFileInput
from feature_engineering import FeatureEngineering
import smnn

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("task_index", 0, "")
flags.DEFINE_string("job_name", 'worker', "distribute tensorflow job.")
flags.DEFINE_string("cluster_conf", 'cluster_conf.json', "distribute config.")
flags.DEFINE_string("model_desc", "dssm_model_desc.json", "model description file")
flags.DEFINE_string("input_schema", "dssm_input_schema.json", "input schema description file")
flags.DEFINE_string("parse_schema", "dssm_parse_schema.json", "config for parse raw data")
flags.DEFINE_string("smnn_schema", "wd_smnn_schema.json", "config for parse raw data")
flags.DEFINE_string("run_function", "train", "train, evaluate, export")
flags.DEFINE_string('metric_host', "127.0.0.1", 'ganglia send data host')
flags.DEFINE_string('metric_group', "no_group", 'ganglia metric group')
flags.DEFINE_string('metric_group_yarn', "no_group", 'ganglia metric group on yarn')
flags.DEFINE_integer("metric_interval", 60, "metric to ganglia interval, unit:seconds")
flags.DEFINE_string('metric_group_for_evaluate', "no_group", 'ganglia metric group for evaluate')
#flags.DEFINE_string('deep_init_dir', "hdfs://in-tensorflow/user/admin/yangsen/hyperparameter/hash_size_v1/checkpoint", 'deep init path')

class DssmEstimator(object):
    def __init__(self, task_index=0, job_name="worker", cluster_conf=None, model_desc=None, input_schema=None, parse_schema=None):
        self.args = smnn.Args()
        self.context = smnn.Context()
        self.context.deserialize_from_args(self.args)
        self._task_index = task_index
        self._job_name = job_name
        self._cluster_conf = cluster_conf
        self._model_desc = model_desc
        self._input_schema = input_schema
        self._parse_schema = parse_schema
        self.smnn_schema = FLAGS.smnn_schema 
        f = open(self._model_desc)
        self._model_desc_content = f.read()
        f.close()
        succ, model_desc_obj = json_util.SafeLoadJsonString(self._model_desc_content)
        if not succ:
            msg = 'parse model json failed'
            print(msg)
        self._model_desc_obj = model_desc_obj
        self._model_parameters = model_desc_obj.get("modelConfig").get("parameters")
        self._is_run_dist = self._model_parameters.get("run_dist")
        self._label = self._model_parameters.get("label")
        self._batch_size = self._model_desc_obj.get("dataSource").get("parameters").get("batch_size")
        if(FLAGS.run_function == "predict"):
            self._batch_size = 1
        self._save_checkpoints_secs = self._model_parameters.get("save_checkpoints_secs")
        self.record_flag = '[ditt]'

    def tensor_dict_from_kafka(self, parameters):
        parameters['need_ckpt'] = True
        if not 'train' in FLAGS.run_function:
            group_id = parameters.get("group_id")
            parameters['group_id'] = group_id + FLAGS.run_function
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
                batch_size=self._batch_size,
                group_id=group_id, 
                seek_to_begin=False,
                seek_to_end=True,
                max_partition_fetch_mb=4,
                need_checkpoint=parameters['need_ckpt'])
        return tensor_dict
    
    def tensor_dict_from_hdfs(self, parameters):
        cycle = parameters.get('cycle', 1)
        is_shuffle = parameters.get('is_shuffle', 0)
        mode = 'train' if 'train' in FLAGS.run_function else 'test'
        data_path = [os.path.join(parameters[mode]['data_path'], f)
                     for f in parameters[mode]['data_path_list']]
        data_path = ','.join(data_path)
        file_filter_pattern = ".*\.(crc|swp)"
        input_uri = '{}?record_flag={}'.format(data_path, self.record_flag)
        tf.logging.info("Get data from file, uri[%s]", input_uri)
        tensor_dict = SmKVRecordFileInput(
                input_uri,
                self.smnn_schema,
                batch_size=self._batch_size,
                checkpoint_dir=self.reader_checkpoint_path,
                need_shuffle=is_shuffle,
                num_epochs=cycle,
                file_filter_pattern=file_filter_pattern)
        return tensor_dict


    def create_reader_new(self):
        source_type = self._model_desc_obj.get("dataSource").get("source_type")
        parameters = self._model_desc_obj.get("dataSource").get("parameters")
        print ('--------- create_reader parameters start ---------')
        for k in sorted(parameters.keys()):
            print (k, parameters[k])
        print ('---------- create_reader parameters end ----------')
        if source_type == "kafka":
            reader = self.tensor_dict_from_kafka(parameters)
        elif source_type == "file":
            reader = self.tensor_dict_from_hdfs(parameters)
        fe = FeatureEngineering()
        reader = fe.get_tensor_dict(reader)
        reader.init(self.context)
        return reader

    def input_fn_train_new(self):
        reader = self.create_reader_new()
        return reader.tensor_dict, tf.shape([-1])

    def input_fn_train(self, reader):
        print("******************* input fn ********************")
        reader.init()
        raw_input_tensor = reader.get_raw_input_tensor()
        print (self._input_schema)
        print (self._parse_schema)
        standard_kv_parser = lib_parser.StandardKvParser(raw_input_tensor,
                                                         self._input_schema,
                                                         self._parse_schema)
        tensor_dict = standard_kv_parser.get_tensor_dict()
        label_tensor = tensor_dict.get(self._label)
        label_reshape = tf.reshape(label_tensor, [self._batch_size,])

        return tensor_dict, label_reshape

    def create_reader(self, num_workers, is_train):
        data_source = self._model_desc_obj.get("dataSource")
        source_type = data_source.get("source_type")
        if source_type == "kafka":
            parameters = data_source.get("parameters")
            bootstrap_servers = parameters.get("bootstrap_servers")
            group_id = parameters.get("group_id")
            if FLAGS.run_function == 'train':
                topic = parameters.get("train_topic")
            elif FLAGS.run_function == 'evaluate':
                topic = parameters.get("test_topic")
            auto_offset_reset = parameters.get("auto_offset_reset")
            batch_size = self._batch_size
            metric_interval = parameters.get("metric_interval", 60)
            auto_metric = parameters.get("auto_metric", True)
            # seek_to_end = parameters.get("seek_to_end", False)
            seek_to_end = parameters.get("seek_to_end")

            print("batch size type: %s" % type(batch_size))
            return standard_kv_reader.StandardKvKafkaReader(bootstrap_servers=bootstrap_servers,
                                                           max_partition_fetch_bytes=1048576,
                                                           group_id=group_id,
                                                           topic=topic,
                                                           auto_offset_reset=auto_offset_reset,
                                                           batch_size=batch_size,
                                                           worker_index=FLAGS.task_index,
                                                           total_workers=num_workers,
                                                           metric_interval=metric_interval,
                                                           auto_metric=auto_metric,
                                                           seek_to_end=seek_to_end,
                                                           is_train=is_train)
        elif source_type == "hdfs":
            parameters = data_source.get("parameters")
            batch_size = self._batch_size
            if FLAGS.run_function == "train":
                data_path = parameters.get("train_data")
                process_file = parameters.get("processed_train_file")
            elif FLAGS.run_function == "evaluate":
                data_path = parameters.get("test_data")
                process_file = parameters.get("processed_test_file")
            elif FLAGS.run_function == "predict":
                data_path = parameters.get("predict_data")
                process_file = parameters.get("processed_test_file")
            begin_time = parameters.get("begin_time")
            dir_depth = parameters.get("dir_depth")
            cycle = parameters.get("cycle", 1)
            parser = schema_parser.KvSchemaParser(FLAGS.input_schema)
            st = parser.parse(FLAGS.parse_schema)
            return standard_kv_reader.StandardKvHdfsReader(data_path,
                                                           batch_size,
                                                           record_flag='[ditt]',
                                                           cycle=cycle,
                                                           worker_index=FLAGS.task_index,
                                                           total_num_worker=num_workers,
                                                           reader_checkpoint=process_file,
                                                           begin_time=begin_time,
                                                           dir_depth=dir_depth)
        else:
            return None

    def gen_run_config(self):
        import json
        cluster = json.load(open(self._cluster_conf, "r"))
        print(cluster)
        cluster_spec = tf.train.ClusterSpec(cluster)
        tf_config = {'cluster': cluster, 'task': {'type': self._job_name, 'index': self._task_index}} #ClusterConfig说明中，task 配置的key为task_id,实际代码中用的task
        if self._is_run_dist:
            os.environ['TF_CONFIG'] = json.dumps(tf_config)
        run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=self._save_checkpoints_secs, keep_checkpoint_max=10)
        #run_config = tf.estimator.RunConfig()
        num_workers = len(cluster['worker'])
        num_ps = len(cluster['ps'])
        print("**********************job base info **********************")
        print("cluster:")
        print(cluster)
        print("task type : %s" % FLAGS.job_name)
        print("task index: %d" % FLAGS.task_index)
        print("ps num : %d, work num : %d" % (num_ps, num_workers))
        print("tf_confg: %s" % tf_config)
        return cluster_spec, run_config, num_ps, num_workers

    def train(self):
        if self._is_run_dist:
            cluster_spec, run_config, num_ps, num_workers = self.gen_run_config()
            server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        else:
            run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=self._save_checkpoints_secs)
            num_workers = 1
            server = None
        if FLAGS.job_name == 'ps':
            print("ps start")
            server.join()
        elif FLAGS.job_name == 'worker':
            print("worker start")
            print("input schema path: %s" % FLAGS.input_schema)
            print("parse schema path: %s" % FLAGS.parse_schema)
            train_steps = self._model_parameters.get("train_steps")
            #reader = self.create_reader(num_workers,True)
            builder = DssmBuilder(data_source_desc=self._input_schema, model_desc=self._model_desc_content)
            model = builder.build_estimator(run_config=run_config)
            #monitors = [StartStandardKvReaderHook(reader)]
            #monitors.append(init_embeddings.InitEmbeddingsHook(FLAGS.deep_init_dir), hook_log)
            print(train_steps)
            model.train(input_fn=lambda: self.input_fn_train_new(), steps=train_steps)

    def add_metrics(self, model):
        def auc(features, labels, predictions):
            return {'auc': tf.metrics.auc(labels, predictions['logistic'])}

        def group_auc(features, labels, predictions):
            recoid = features["reco_id"]
            return {'group_auc': custom_metric.group_auc('group_auc', predictions['logistic'], labels, recoid, "_base")}

        def app_iflow_auc(features, labels, predictions):
            app = features["app"]
            return {'app_iflow_auc': custom_metric.cond_auc(predictions['logistic'], labels, app, "app_iflow", "app_iflow_auc")}
        
        def browser_iflow_auc(features, labels, predictions):
            app = features["app"]
            return {'browser_iflow_auc': custom_metric.cond_auc(predictions['logistic'], labels, app, "browser_iflow", "browser_iflow_auc")}

        def browser_rec_iflow_auc(features, labels, predictions):
            batch_size = array_ops.shape(predictions['logistic'])[0]
            app = tf.reshape(features["app"], [batch_size,])
            item_type = tf.reshape(features["item_type"], [batch_size,])
            is_top = tf.reshape(features["is_top"], [batch_size,])
            recoid = features["reco_id"]
            app_mask = tf.equal("browser_iflow",app)
            item_type_mask = tf.not_equal("208" ,item_type)
            is_top_mask = tf.not_equal("1" , is_top)
            merge_mask = app_mask & item_type_mask & is_top_mask
            return {"browser_rec_iflow_auc": custom_metric.group_auc('browser_rec_iflow_auc',
                    tf.boolean_mask(predictions['logistic'], merge_mask),
                    tf.boolean_mask(labels, merge_mask), 
                    tf.boolean_mask(recoid, merge_mask), "_base")}

        model = tf.contrib.estimator.add_metrics(model, auc)
        model = tf.contrib.estimator.add_metrics(model, group_auc)
        model = tf.contrib.estimator.add_metrics(model, browser_iflow_auc)
        model = tf.contrib.estimator.add_metrics(model, browser_rec_iflow_auc)

        return model
    
    def evaluate(self):
        if self._model_desc_obj.get("dataSource").get("source_type") == "kafka":
            group_id = self._model_desc_obj.get("dataSource").get("parameters").get("group_id")
            self._model_desc_obj["dataSource"]["parameters"]["group_id"] = group_id + "_eval"
            self._model_desc_obj["dataSource"]["parameters"]["seek_to_end"] = True
            self._model_desc_obj["dataSource"]["parameters"]["auto_metric"] = False
        n_class = self._model_parameters.get("n_classes", 2)
        test_steps = self._model_parameters.get("test_steps")
        reader = self.create_reader(1,False)
        builder = DssmBuilder(data_source_desc=self._input_schema, model_desc=self._model_desc_content)
        model = builder.build_estimator()
        model = self.add_metrics(model)
        monitors = [StartStandardKvReaderHook(reader)]
        results = model.evaluate(input_fn=lambda: self.input_fn_train(reader),  hooks=monitors, steps=test_steps)
        print(results)
        status_tool.report_metric(results, results['global_step'])

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

    def generic_signature_fn(self, examples, unused_features, predictions):
        if examples is None:
            raise ValueError('examples cannot be None when using this signature fn.')
        named_graph_signature = {
            'inputs': exporter.generic_signature({'inputs': examples}),
            'outputs': exporter.generic_signature({'prob': predictions})
        }
        return {}, named_graph_signature

    def serving_input_fn(self):
        examples = tf.placeholder(dtype=tf.string, shape=[None])
        standard_kv_parser = lib_parser.StandardKvParser(examples, self._input_schema, self._parse_schema)
        tensor_dict = standard_kv_parser.get_tensor_dict()
        receiver_tensors = {'inputs': examples}
        return tf.estimator.export.ServingInputReceiver(tensor_dict, receiver_tensors)

    def clean_expired_models(self, model_root, reserve_count):
        import logging
        logger = logging.getLogger()
        ''' 
        :exception: tf.errors.OpError
        '''
        try:
            dirnames = tf.gfile.ListDirectory(model_root)
            digit_list = [d for d in dirnames if d.isdigit()]
            dirnames = sorted(digit_list, key = lambda x: int(x), reverse = True)
            need_remove = dirnames[reserve_count:]

            if not need_remove:
                logger.info('Total [%d] model in root[%s], less than threshold[%d]. No need to clean.'\
                                %(len(dirnames), model_root, reserve_count))
            else:
                logger.info('Total [%d] model in root[%s], more than threshold[%d]. Clean expired models.'\
                                %(len(dirnames), model_root, reserve_count))

            for dirname in need_remove:
                full_path = os.path.join(model_root, dirname)
                logger.info('Remove expired dir[%s]' % full_path)
                tf.gfile.DeleteRecursively(full_path)
        except tf.errors.OpError, e:
            msg = 'Remove expired model from [%s] failed. reason: %s' %(dst_model_root, e)
            logger.info('clean_expired_models error, error msg = [%s]' % msg)


    def export(self):
        export_dir = self._model_parameters.get("export_path")
        print("export dir: " + export_dir)
        builder = DssmBuilder(data_source_desc=self._input_schema, model_desc=self._model_desc_content)
        model = builder.build_estimator()
        true_path = model.export_savedmodel(
            export_dir_base=export_dir,
            serving_input_receiver_fn=self.serving_input_fn,
            as_text=True
        )
        print ("export_dir: " + export_dir)
        print ("true_path: " + true_path)
        print ('*' * 40)
        self.clean_expired_models(export_dir, 60)

    def predict(self):
        builder = DssmBuilder(data_source_desc=self._input_schema, model_desc=self._model_desc_content)
        model = builder.build_estimator()
        reader = self.create_reader(1,False)
        hook_log = tf.train.LoggingTensorHook(['dnn_1/input_from_feature_columns/input_from_feature_columns/concat:0','Const:0'], 5)
        monitors = [StartStandardKvReaderHook(reader), hook_log]
        results = model.predict(input_fn=lambda: self.input_fn_train(reader),  hooks=monitors)
        for result in results:
            print(result)


def main(_):
    try:
        estimator = DssmEstimator(FLAGS.task_index, FLAGS.job_name, FLAGS.cluster_conf, FLAGS.model_desc,
                                         FLAGS.input_schema, FLAGS.parse_schema)
        if FLAGS.run_function == "train":
            gmetric_writer.init(job_name=FLAGS.job_name, task_index=FLAGS.task_index, host=FLAGS.metric_host, group=FLAGS.metric_group)
            estimator.train()
        elif FLAGS.run_function == "evaluate":
            gmetric_writer.init(host=FLAGS.metric_host, group=FLAGS.metric_group_yarn)
            estimator.evaluate()
        elif FLAGS.run_function == "export":
            estimator.export()
        elif FLAGS.run_function == "predict":
            estimator.predict()
        else:
            print("run function error, must train, evaluate or export")
            tf.app.run()
    except Exception as e:
        print ("dssm stop")
        traceback.print_exc()
    except:
        print ("dssm stop")
    exit(0)

if __name__ == "__main__":
    tf.app.run()

