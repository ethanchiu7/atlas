#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import six
import math
import collections
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn
from tensorflow.contrib import layers
from tensorflow.python.ops import math_ops
from tensorflow.python.training import ftrl
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.summary import summary
from tensorflow.python.training import adagrad
from tensorflow.python.training import adadelta
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops import variable_scope
from tensorflow.python.estimator import model_fn
from tensorflow.python.ops import control_flow_ops
from export_output import RankClassifierExportOutput
from tensorflow.python.training import training_util
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.estimator.export import export_output
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.feature_column import feature_column as feature_column_lib

_DNN_LEARNING_RATE = 0.08

LossAndLabels = collections.namedtuple('LossAndLabels',
                                       ['unweighted_loss', 'processed_labels'])

def add_hidden_layer_summary(value, tag):
    tf.summary.scalar("%s/fraction_of_zero_values" % tag,
                      nn.zero_fraction(value))
    tf.summary.histogram("%s/activation" % tag, value)

def compute_weighted_loss(loss_unweighted, weights, name="loss"):
    with ops.name_scope(name, values=(loss_unweighted, weights)) as name_scope:
        if weights is None:
            loss = math_ops.reduce_mean(loss_unweighted, name=name_scope)
            return loss, loss
        weights = weights_broadcast_ops.broadcast_weights(weights, loss_unweighted)
        with ops.name_scope(None, "weighted_loss",
                                                (loss_unweighted, weights)) as name:
            weighted_loss = math_ops.multiply(loss_unweighted, weights, name=name)
        weighted_loss_mean = math_ops.reduce_mean(weighted_loss, name=name_scope)
        weighted_loss_normalized = math_ops.div(
                math_ops.reduce_sum(weighted_loss),
                math_ops.to_float(math_ops.reduce_sum(weights)),
                name="weighted_average_loss")
        return weighted_loss_mean, weighted_loss_normalized

class ModelBase:
    def __init__(self, features, mode, params,
                 dnn_parent_scope,
                 logits_dimension=32):
        self.mode = mode
        self.params = params
        self.features = features
        self.dnn_features_columns = params.get('deep')
        self.item_id_column = params.get('item_id')
        self.bias_columns = params.get('bias')
        self.weight_column = params.get('weight_column')
        self.num_ps_replicas = params.get('num_ps_replicas')
        self.dnn_hidden_units = params.get('dnn_hidden_units')
        self.metric_prefix = params.get('metric_prefix')
        self.logits_dimension = logits_dimension
        self.dnn_parent_scope = dnn_parent_scope
        self.graph = tf.get_default_graph()
    
    def init(self):
        self.get_opt()
        self.build_graph()
    
    def init_predict(self):
        self.build_user_graph()

    def get_opt(self):
        self.dnn_optimizer = adagrad.AdagradOptimizer(
                learning_rate=_DNN_LEARNING_RATE)
        #self.dnn_optimizer = tf.train.MomentumOptimizer(_DNN_LEARNING_RATE, 0.8)
 
    def build_user_graph(self):
        self.logits_dnn = self.build_user_dnn()
        logits = self.logits_dnn
        self.logits_combine = ops.convert_to_tensor(logits, name="logits")
   
    def build_graph(self):
        self.logits_dnn = self.build_dnn()
        logits = self.logits_dnn
        self.logits_combine = ops.convert_to_tensor(logits, name="logits")
    
    def build_dnn(self):
        dnn_partitioner = (
                partitioned_variables.min_max_variable_partitioner(
                        max_partitions=self.num_ps_replicas,
                        min_slice_size=64 << 20))
        with tf.variable_scope(self.dnn_parent_scope,
                               values=tuple(six.itervalues(self.features)),
                               partitioner=dnn_partitioner,
                               reuse=False):
            self.get_input_from_feature_columns()
            return self.build_dnn_logits(
                    self.dnn_net,
                    self.dnn_hidden_units,
                    self.logits_dimension,
                    self.dnn_parent_scope) 
 
    def build_user_dnn(self):
        dnn_partitioner = (
                partitioned_variables.min_max_variable_partitioner(
                        max_partitions=self.num_ps_replicas,
                        min_slice_size=64 << 20))
        with tf.variable_scope(self.dnn_parent_scope,
                               values=tuple(six.itervalues(self.features)),
                               partitioner=dnn_partitioner,
                               reuse=False):
            self.get_user_input_from_feature_columns()
            return self.build_dnn_logits(
                    self.dnn_net,
                    self.dnn_hidden_units,
                    self.logits_dimension,
                    self.dnn_parent_scope) 
 
    def get_user_input_from_feature_columns(self):
        input_layer_partitioner = (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=self.num_ps_replicas,
                min_slice_size=64 << 20))
        with variable_scope.variable_scope(
                "input_from_feature_columns",
                values=tuple(six.itervalues(self.features)),
                partitioner=input_layer_partitioner) as dnn_input_scope:
            self.dnn_net = \
                        layers.input_from_feature_columns(
                                columns_to_tensors=self.features,
                                feature_columns=self.dnn_features_columns,
                                weight_collections=[self.dnn_parent_scope],
                                scope=dnn_input_scope)
    
    def get_input_from_feature_columns(self):
        input_layer_partitioner = (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=self.num_ps_replicas,
                min_slice_size=64 << 20))
        with variable_scope.variable_scope(
                "input_from_feature_columns",
                values=tuple(six.itervalues(self.features)),
                partitioner=input_layer_partitioner) as dnn_input_scope:
            self.dnn_net = \
                        layers.input_from_feature_columns(
                                columns_to_tensors=self.features,
                                feature_columns=self.dnn_features_columns,
                                weight_collections=[self.dnn_parent_scope],
                                scope=dnn_input_scope)
            self.item_id_net = \
                        layers.input_from_feature_columns(
                                columns_to_tensors=self.features,
                                feature_columns=self.item_id_column,
                                weight_collections=[self.dnn_parent_scope],
                                scope=dnn_input_scope)
            self.bias_net = \
                        layers.input_from_feature_columns(
                                columns_to_tensors=self.features,
                                feature_columns=self.bias_columns,
                                weight_collections=[self.dnn_parent_scope],
                                scope=dnn_input_scope)
    
    @staticmethod
    def build_dnn_logits(net,
                         dnn_hidden_units,
                         logits_dimension,
                         dnn_parent_scope):
        def residual_fn(nets):
            return tf.add_n(nets)

        layers_list = []
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with variable_scope.variable_scope(
                    "hiddenlayer_%d" % layer_id,
                    values=(net,)) as dnn_hidden_layer_scope:
                net = layers.fully_connected(
                        net,
                        num_hidden_units,
                        activation_fn=nn.elu,
                        variables_collections=[dnn_parent_scope],
                        scope=dnn_hidden_layer_scope)
                net = residual_fn([net] + layers_list)
                layers_list.append(net)
                add_hidden_layer_summary(net, dnn_hidden_layer_scope.name)
        with variable_scope.variable_scope(
                "logits", values=(net,)) as dnn_logits_scope:
            dnn_net = layers.fully_connected(
                    net,
                    logits_dimension,
                    activation_fn=None,
                    variables_collections=[dnn_parent_scope],
                    scope=dnn_logits_scope)
            add_hidden_layer_summary(dnn_net, dnn_logits_scope.name)
            return dnn_net
    
    def train_op_fn(self, training_loss):
        train_ops = []
        global_step = training_util.get_global_step()
        if self.logits_dnn is not None:
            var_list = ops.get_collection(
                    ops.GraphKeys.TRAINABLE_VARIABLES,
                    scope=self.dnn_parent_scope)
            train_op = self.dnn_optimizer.minimize(
                    training_loss, var_list=var_list)
            train_ops.append(train_op)
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)
    
    def calc_weight(self):
        with ops.name_scope(None, 'weights', values=self.features.values()):
            if self.weight_column is None or self.mode == model_fn.ModeKeys.PREDICT or  self.mode == model_fn.ModeKeys.EVAL:
                return 1.
            weight_column = feature_column_lib.numeric_column(key=self.weight_column)
            weights = weight_column._get_dense_tensor(feature_column_lib._LazyBuilder(self.features))
            weights = math_ops.to_float(weights, name='weights')
            return weights
            
    def calc_loss(self, logits):
        unweighted_loss, processed_labels = self.create_loss(logits)
        weights = self.calc_weight()
        training_loss, _metric_loss = compute_weighted_loss(
                unweighted_loss,
                weights=weights)
        return unweighted_loss, processed_labels, training_loss, weights
    
    def train_fn(self, training_loss, unweighted_loss, weights, predictions):
        with ops.name_scope(''):
            summary.scalar('loss', training_loss)
            average_loss = losses.compute_weighted_loss(
                    unweighted_loss,
                    weights=weights,
                    reduction=losses.Reduction.MEAN)
            summary.scalar('average_loss', average_loss)
        return model_fn.EstimatorSpec(
                mode=model_fn.ModeKeys.TRAIN,
                predictions=predictions,
                loss=training_loss,
                train_op=self.train_op_fn(training_loss))
    
class DSSM(ModelBase):
    def __init__(self, features, mode, params, dnn_parent_scope='dnn'):
        ModelBase.__init__(self, features, mode, params, dnn_parent_scope)

    def create_loss(self, logits):
        labels = math_ops.to_float(self.features['read'])
        return LossAndLabels(
                unweighted_loss=nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=logits),
                processed_labels=labels)
    
    def create_predict_estimator_spec(self):
        self.item_dir = self.params['predict_item_dir'] 
        self.langs =  self.params['predict_langs']
        user_lang = self.features['ct_lang'][0]
        utdid = self.features['utdid']
        user_embed = self.logits_combine
        table_ref = None
        for lang in self.langs:
            table_lang = self.get_table(lang)
            if table_ref is None:
                table_ref = table_lang
            else:
                table_ref = tf.cond(tf.reshape(tf.equal(user_lang, lang),[]),lambda: table_lang, lambda: table_ref)
        result = tf.user_ops.flann_table_find(table_ref, user_embed, k=380, checks=-1, Tout=tf.int64)
        if isinstance(result, tuple):
            items, scores = result
            predictions = {
                    'outputs': items,
                    'scores': -scores
                    }
        else:
            items = result
            predictions = {
                    'outpus': items
                    }
        return model_fn.EstimatorSpec(
                mode=model_fn.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={'predict': export_output.PredictOutput(predictions)})


    def read_items_dict(self, lang, data_dir=None):
        path = os.path.join(data_dir, lang)
        items_arr = np.loadtxt(path, dtype=np.str, delimiter=' ', comments='####')
        items_dict = {
            "item_id": tf.constant(items_arr[:, 0])
        }
        return items_dict

    def get_table(self, lang):
        print("Start to read item from %s, lang[%s]" % (self.item_dir, lang))
        items_dict = self.read_items_dict(lang, self.item_dir)

        dnn_partitioner = (
                partitioned_variables.min_max_variable_partitioner(
                        max_partitions=self.num_ps_replicas,
                        min_slice_size=64 << 20))
        with tf.variable_scope(self.dnn_parent_scope,
                               values=tuple(six.itervalues(self.features)),
                               partitioner=dnn_partitioner,
                               reuse=tf.AUTO_REUSE):
            input_layer_partitioner = (
                partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.num_ps_replicas,
                    min_slice_size=64 << 20))
            with variable_scope.variable_scope(
                    "input_from_feature_columns",
                    values=tuple(six.itervalues(self.features)),
                    partitioner=input_layer_partitioner) as dnn_input_scope:
                    item_id_net = layers.input_from_feature_columns(
                                    columns_to_tensors= items_dict,
                                    feature_columns=self.item_id_column,
                                    weight_collections=[self.dnn_parent_scope],
                                    scope=dnn_input_scope)
        item_id_int = tf.string_to_number(items_dict['item_id'], out_type=tf.int64)
        table_ref = tf.user_ops.flann_table(key_dtype=tf.float32, value_dtype=tf.int64)
        build_index = tf.user_ops.flann_table_insert(
                table_ref, item_id_net, item_id_int,
                algorithm=12, far_sample_n=3)
        self.graph.add_to_collection(
                tf.GraphKeys.TABLE_INITIALIZERS, build_index)
        return table_ref

    def create_estimator_spec(self):
        # Predict.
        with ops.name_scope('head'):
            with ops.name_scope(None, 'predictions', (self.logits_combine,)):
                dnn_logits = head_lib._check_logits(self.logits_combine, self.logits_dimension)
                item_id_net = self.item_id_net
                if(self.mode == model_fn.ModeKeys.EVAL):
                    logits = tf.reduce_sum(tf.multiply(dnn_logits, item_id_net), reduction_indices=1, keep_dims = True)
                    #logits = tf.reduce_sum(tf.multiply(dnn_logits, item_id_net), reduction_indices=1, keep_dims = True) + tf.reduce_sum(self.bias_net, axis = 1, keep_dims = True)
                else:
                    logits = tf.reduce_sum(tf.multiply(dnn_logits, item_id_net), reduction_indices=1, keep_dims = True) + tf.reduce_sum(self.bias_net, axis = 1, keep_dims = True)
                logistic = math_ops.sigmoid(logits, name='logistic')
                two_class_logits = array_ops.concat(
                        (array_ops.zeros_like(logits), logits), 1,
                        name='two_class_logits')
                scores = nn.softmax(two_class_logits, name='probabilities')
                class_ids = array_ops.reshape(
                        math_ops.argmax(two_class_logits, axis=1),
                        (-1, 1),
                        name='classes')
                classes = string_ops.as_string(class_ids, name='str_classes')
                predictions = {
                    'logits': logits,
                    'logistic': logistic,
                    'probabilities': scores,
                    'class_ids': class_ids,
                    'classes': classes,
                }
            
            # calculate loss
            unweighted_loss, processed_labels, training_loss, weights = self.calc_loss(logits)
            
            # Eval.
            if self.mode == model_fn.ModeKeys.EVAL:
                return model_fn.EstimatorSpec(
                        mode=model_fn.ModeKeys.EVAL,
                        predictions=predictions,
                        loss=training_loss)
            # Train
            return self.train_fn(training_loss, unweighted_loss, weights, predictions)


def self_model_fn(features, labels, mode, params):
    del labels # no use
    model = DSSM(features, mode, params)
    if(mode != model_fn.ModeKeys.PREDICT):
        model.init()
        return model.create_estimator_spec()
    else:
        model.init_predict()
        return model.create_predict_estimator_spec()

if __name__ == "__main__":
    pass

