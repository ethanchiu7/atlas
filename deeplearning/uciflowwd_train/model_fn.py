#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import six
import math
import collections
import tensorflow as tf
from metric import Metric
from seq_to_vec import SeqModelSimple
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
from tensorflow.contrib.layers.python.layers import feature_column_ops
from tensorflow.contrib.layers.python.layers import feature_column as fc
import custom_metric 

_DNN_LEARNING_RATE = 0.05
_SEQ_LEARNING_RATE = 0.1
_LINEAR_LEARNING_RATE = 0.2

LossAndLabels = collections.namedtuple('LossAndLabels',
                                       ['unweighted_loss', 'processed_labels'])

def _linear_learning_rate(num_linear_feature_columns):
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(_LINEAR_LEARNING_RATE, default_learning_rate)

def add_hidden_layer_summary(value, tag):
    tf.summary.scalar("%s/fraction_of_zero_values" % tag,
                      nn.zero_fraction(value))
    tf.summary.histogram("%s/activation" % tag, value)

def add_logits_summary(value, tag):
    tf.summary.scalar('avg_%s' % tag, tf.reduce_mean(value))

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

def input_from_feature_columns(columns_to_tensors,
                               feature_columns,
                               weight_collections,
                               scope,
                               trainable=True,
                               output_rank=2,
                               default_name='input_from_feature_columns'):
    columns_to_tensors = columns_to_tensors.copy()
    feature_column_ops.check_feature_columns(feature_columns)
    with variable_scope.variable_scope(scope,
                                       default_name=default_name,
                                       values=columns_to_tensors.values()):
        output_tensors_dict = {}
        transformer = feature_column_ops._Transformer(columns_to_tensors)
        if weight_collections:
            weight_collections = list(set(list(weight_collections) +
                                          [ops.GraphKeys.GLOBAL_VARIABLES]))
        for column in sorted(set(feature_columns), key=lambda x: x.key):
            with variable_scope.variable_scope(None,
                                               default_name=column.name,
                                               values=columns_to_tensors.values()):
                transformed_tensor = transformer.transform(column)
                key = column.key
                try:
                    # pylint: disable=protected-access
                    arguments = column._deep_embedding_lookup_arguments(
                            transformed_tensor)
                    output_tensors_dict[key] = \
                            fc._embeddings_from_arguments(    # pylint: disable=protected-access
                                    column,
                                    arguments,
                                    weight_collections,
                                    trainable,
                                    output_rank=output_rank)
                except NotImplementedError as ee:
                    try:
                        # pylint: disable=protected-access
                        output_tensors_dict[key] = \
                                column._to_dnn_input_layer(
                                        transformed_tensor,
                                        weight_collections,
                                        trainable,
                                        output_rank=output_rank)
                    except ValueError as e:
                        raise ValueError('Error creating input layer for column: {}.\n'
                                                         '{}, {}'.format(column.name, e, ee))
        return output_tensors_dict

class DNNLinearCombinedClassifier():
    def __init__(self, features, labels, mode, params, logits_dimension=1):
        self.mode = mode
        self.params = params
        self.features = features
        self.labels = labels
        self.dnn_features_columns = params.get('deep')
        self.linear_feature_columns = params.get('wide')
        self.weight_column = params.get('weight_column')
        self.num_ps_replicas = params.get('num_ps_replicas')
        self.dnn_hidden_units = params.get('dnn_hidden_units')
        self.metric_prefix = params.get('metric_prefix')
        self.logits_dimension = logits_dimension
        self.dnn_parent_scope = 'dnn'
        self.linear_parent_scope = 'linear'
    
    def init(self):
        self.get_opt()
        self.build_graph()
    
    def get_opt(self):
        self.dnn_optimizer = adagrad.AdagradOptimizer(
                learning_rate=_DNN_LEARNING_RATE)
        self.seq_optimizer = adagrad.AdagradOptimizer(
            learning_rate=_SEQ_LEARNING_RATE)
        self.linear_optimizer = ftrl.FtrlOptimizer(
                #learning_rate=_linear_learning_rate(len(self.linear_feature_columns)),
                learning_rate=_LINEAR_LEARNING_RATE,
                learning_rate_power=-0.5,
                initial_accumulator_value=0.1,
                l1_regularization_strength=3.0,
                l2_regularization_strength=5.0)
    
    def build_graph(self):
        self.logits_dnn = self.build_dnn()
        self.logits_linear = self.build_linear_logits(
                self.features,
                self.linear_feature_columns,
                self.num_ps_replicas,
                self.logits_dimension,
                self.linear_parent_scope)
        logits = self.logits_dnn + self.logits_linear
        self.logits_combine = ops.convert_to_tensor(logits, name="logits")
        add_logits_summary(self.logits_dnn, "logits_dnn")
        add_logits_summary(self.logits_linear, "logits_linear")
        add_logits_summary(self.logits_combine, "logits_combine")
    
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
            self.get_features()
            return self.build_dnn_logits(
                    self.dnn_features,
                    self.dnn_hidden_units,
                    self.logits_dimension,
                    self.dnn_parent_scope)
    
    def get_input_from_feature_columns(self):
        input_layer_partitioner = (
            partitioned_variables.min_max_variable_partitioner(
                max_partitions=self.num_ps_replicas,
                min_slice_size=64 << 20))
        with variable_scope.variable_scope(
                "input_from_feature_columns",
                values=tuple(six.itervalues(self.features)),
                partitioner=input_layer_partitioner) as dnn_input_scope:
            self.dnn_features = input_from_feature_columns(
                                columns_to_tensors=self.features,
                                feature_columns=self.dnn_features_columns,
                                weight_collections=[self.dnn_parent_scope],
                                scope=dnn_input_scope)
    
    def get_features(self):
        # 抽取高级特征
        self._seq2vec_scope = "seq"
        model_params = {
            "seq_model_scope": self._seq2vec_scope,
            "num_ps_replicas": self.num_ps_replicas,
            "itemId_hash_bucket_size": int(1e7),
            "itemId_embedding_size": 32,
            "hidden_units": [16,],
        }
        self._seq2vec_model = SeqModelSimple(
            self.features, self.mode,
            params=model_params)
        vectors = self._seq2vec_model.calc_vectors()
        self.dnn_features.update(vectors)
    
    @staticmethod
    def build_dnn_logits(dnn_features,
                         dnn_hidden_units,
                         logits_dimension,
                         dnn_parent_scope):
        tensors = []
        for key in sorted(dnn_features.keys()):
            output_tensor = dnn_features[key]
            tensors.append(dnn_features[key])
            tf.logging.info('[build_dnn_logits] key={}, tensor={}'.format(key, output_tensor))
        net = tf.concat(tensors, axis=1)
        for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
            with variable_scope.variable_scope(
                    "hiddenlayer_%d" % layer_id,
                    values=(net,)) as dnn_hidden_layer_scope:
                net = layers.fully_connected(
                        net,
                        num_hidden_units,
                        activation_fn=nn.relu,
                        variables_collections=[dnn_parent_scope],
                        scope=dnn_hidden_layer_scope)
                add_hidden_layer_summary(net, dnn_hidden_layer_scope.name)
        with variable_scope.variable_scope(
                "logits", values=(net,)) as dnn_logits_scope:
            dnn_logits = layers.fully_connected(
                    net,
                    logits_dimension,
                    activation_fn=None,
                    variables_collections=[dnn_parent_scope],
                    scope=dnn_logits_scope)
            add_hidden_layer_summary(dnn_logits, dnn_logits_scope.name)
            return dnn_logits
    
    @staticmethod
    def build_linear_logits(features,
                            linear_feature_columns,
                            num_ps_replicas,
                            logits_dimension,
                            linear_parent_scope):
        linear_partitioner = partitioned_variables.min_max_variable_partitioner(
                max_partitions=num_ps_replicas,
                min_slice_size=64 << 20)
        with variable_scope.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=linear_partitioner) as scope:
            linear_logits, _, _ = layers.weighted_sum_from_feature_columns(
                    columns_to_tensors=features,
                    feature_columns=linear_feature_columns,
                    num_outputs=logits_dimension,
                    weight_collections=[linear_parent_scope],
                    scope=scope)
        return linear_logits
    
    def train_op_fn(self, training_loss, incr_global_step=True):
        summaries = ['gradients', 'gradient_norm']
        train_ops = []
        global_step = training_util.get_global_step()
        if self.logits_dnn is not None:
            #var_list = ops.get_collection(
            #        ops.GraphKeys.TRAINABLE_VARIABLES,
            #        scope=self.dnn_parent_scope)
            #train_op = self.dnn_optimizer.minimize(
            #        training_loss, var_list=var_list)
            train_op = optimizers.optimize_loss(
                    loss=training_loss,
                    global_step=global_step,
                    learning_rate=_DNN_LEARNING_RATE,
                    optimizer=self.dnn_optimizer,
                    variables=(ops.get_collection(self.dnn_parent_scope)
                               +ops.get_collection(self._seq2vec_scope)),
                    name=self.dnn_parent_scope,
                    summaries=summaries,
                    increment_global_step=False)
            train_ops.append(train_op)
        if self.logits_linear is not None:
            #var_list = ops.get_collection(
            #        ops.GraphKeys.TRAINABLE_VARIABLES,
            #        scope=self.linear_parent_scope)
            #train_op = self.linear_optimizer.minimize(
            #        training_loss, var_list=var_list)
            train_op = optimizers.optimize_loss(
                    loss=training_loss,
                    global_step=global_step,
                    learning_rate=_linear_learning_rate(len(self.linear_feature_columns)),
                    optimizer=self.linear_optimizer,
                    variables=ops.get_collection(self.linear_parent_scope),
                    name=self.linear_parent_scope,
                    summaries=summaries,
                    increment_global_step=False)
            train_ops.append(train_op)
        train_op = control_flow_ops.group(*train_ops)
        if not incr_global_step:
            return train_op
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1)
    
    def calc_weight(self):
        with ops.name_scope(None, 'weights', values=self.features.values()):
            if self.weight_column is None or self.mode == model_fn.ModeKeys.PREDICT or  self.mode == model_fn.ModeKeys.EVAL:
                return 1.

            weight = tf.ones_like(self.features["read_tm_vl"])
	        #阅读时长提权[1,2]
            weight_t = self.features["read_tm_vl"]
            weight_t = math_ops.to_float(weight_t, name='weights') / 1000.0
            weight_t = math_ops.log(tf.maximum(1.0, 1.0 + weight_t)) / math_ops.log(2.0)
            weight1 = tf.maximum(4.0,tf.minimum(7.5, weight_t))
            #正样本提权
            weight_p = weight * 2.0
            weight_n = weight
            weight2 = tf.where(tf.greater(self.features['read'],0),weight_p,weight_n)
            weight3 = weight1 * weight2
            return weight3
            
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
    
    def add_metric_prefix(self, metric_ops):
        if self.metric_prefix:
            for k in metric_ops.keys():
                metric_ops[self.metric_prefix + k] = metric_ops.pop(k)
        return metric_ops

    def custom_metric(self,name,pred,labels,group=None,suffix="",config=None):
        return custom_metric.custom_metric(name,pred,labels,group,suffix,config)
    
    def eval_metric_ops(self,
                        labels,
                        logits,
                        logistic,
                        scores,
                        class_ids,
                        unweighted_loss,
                        weights=None):
        with ops.name_scope(None, 'metrics', (labels, logits, logistic, scores,
                  class_ids, unweighted_loss, weights)):
            labels_mean = Metric.indicator_labels_mean(
                    labels=labels, weights=weights, name='label/mean')
            batch_size = array_ops.shape(logits)[0]
            label_reshape = tf.reshape(labels, [batch_size,])
            pred_reshape = tf.reshape(scores[:,-1], [batch_size,])
            item_id_reshape = tf.reshape(self.features['item_id'], [batch_size,])
            app_reshape = tf.reshape(self.features['app'], [batch_size,])
            #score_reshape = tf.reshape(self.features['score'], [batch_size,])
    	    utdid_reshape = tf.reshape(self.features["utdid"], [batch_size,])
            app_mask = tf.equal("app_iflow",app_reshape)
            browser_mask = tf.equal("browser_iflow",app_reshape)
            score_mask = tf.greater_equal(pred_reshape, 0.05)

            metric_ops = {
                    'average_loss': metrics_lib.mean(
                            unweighted_loss,
                            weights=weights,
                            name='average_loss'),
                    'accuracy': metrics_lib.accuracy(
                            labels=labels,
                            predictions=class_ids,
                            weights=weights,
                            name='accuracy'),
                    'prediction/mean': Metric.predictions_mean(
                            predictions=logistic,
                            weights=weights,
                            name='prediction/mean'),
                    'label/mean': labels_mean,
                    'accuracy_baseline': Metric.accuracy_baseline(labels_mean),
                    'auc': Metric.auc(
                            labels=labels,
                            predictions=logistic,
                            weights=weights,
                            name='auc'),
                    'auc_precision_recall': Metric.auc(
                            labels=labels,
                            predictions=logistic,
                            weights=weights,
                            curve='PR',
                            name='auc_precision_recall'),
			"auc_utdid_app": self.custom_metric("group_auc",tf.boolean_mask(pred_reshape,app_mask),tf.boolean_mask(label_reshape,app_mask),tf.boolean_mask(utdid_reshape,app_mask),"_utdid_app"),
			"auc_utdid": self.custom_metric("group_auc",pred_reshape,label_reshape,utdid_reshape,"_utdid"),
			"auc_browser": Metric.auc_cond(pred_reshape,label_reshape,app_reshape,"browser_iflow","auc_browser"),
			"auc_app": Metric.auc_cond(pred_reshape,label_reshape,app_reshape,"app_iflow","auc_app"),
            }
            return self.add_metric_prefix(metric_ops)
    
    def create_loss(self, logits):
        labels = math_ops.to_float(self.features['read'])
        #labels = math_ops.to_float(self.labels)
        return LossAndLabels(
                unweighted_loss=nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=logits),
                processed_labels=labels)
    
    def create_estimator_spec(self):
        # Predict.
        with ops.name_scope('head'):
            with ops.name_scope(None, 'predictions', (self.logits_combine,)):
                logits = head_lib._check_logits(self.logits_combine, self.logits_dimension)
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
            
            if self.mode == model_fn.ModeKeys.PREDICT:
                #batch_size = array_ops.shape(logistic)[0]
                #export_class_list = string_ops.as_string([0, 1])
                #export_output_classes = array_ops.tile(
                #        input=array_ops.expand_dims(input=export_class_list, axis=0),
                #        multiples=[batch_size, 1])
                classifier_output = RankClassifierExportOutput(prob=scores)
                return model_fn.EstimatorSpec(
                        mode=model_fn.ModeKeys.PREDICT,
                        predictions=predictions,
                        export_outputs={
                            'serving_default': classifier_output,
                            'classification': classifier_output,
                            'regression': export_output.RegressionOutput(logistic),
                            'predict': export_output.PredictOutput(predictions)
                        })
            
            # calculate loss
            unweighted_loss, processed_labels, training_loss, weights = self.calc_loss(logits)
            
            # Eval.
            if self.mode == model_fn.ModeKeys.EVAL:
                eval_metric_ops = self.eval_metric_ops(
                        labels=processed_labels,
                        logits=logits,
                        logistic=logistic,
                        scores=scores,
                        class_ids=class_ids,
                        unweighted_loss=unweighted_loss,
                        weights=weights)
                return model_fn.EstimatorSpec(
                        mode=model_fn.ModeKeys.EVAL,
                        predictions=predictions,
                        loss=training_loss,
                        eval_metric_ops=eval_metric_ops)
            # Train
            return self.train_fn(training_loss, unweighted_loss, weights, predictions)

def feature_normalization(features):
    features['ToCtr'] = tf.where(tf.greater(features['ToPv'], 500.),
                                 features['ToCtr'],
                                 tf.ones_like(features['ToCtr']) / 10.)
    features['PreCtr'] = tf.where(tf.greater(features['PrePv'], 500.),
                                 features['PreCtr'],
                                 tf.ones_like(features['PreCtr']) / 10.)
    features['rt_ctr'] = tf.where(tf.greater(features['rt_pv'], 500.),
                                 features['rt_ctr'],
                                 tf.ones_like(features['rt_ctr']) / 10.)


def self_model_fn(features, labels, mode, params):
    #feature_normalization(features)
    #del labels # no use
    model = DNNLinearCombinedClassifier(features, labels, mode, params)
    model.init()
    return model.create_estimator_spec()

if __name__ == "__main__":
    pass

