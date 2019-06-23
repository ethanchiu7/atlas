#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.contrib.metrics.python.ops import metric_ops
from sklearn.metrics import roc_auc_score
class _GroupMetric:
    def __init__(self):
        self.labels = []
        self.groups = []
        self.predictions = []
        self.group_ids = np.array([], dtype=int)
    def add_data(self, label, prediction, group, group_id):
        self.predictions = np.append(self.predictions, prediction)
        self.labels = np.append(self.labels, label)
        self.groups = np.append(self.groups, group)
        self.group_ids = np.append(self.group_ids, group_id)
    def calc_group_mse(self):
        def func(df):
            return np.abs(np.mean(df['labels']) - np.mean(df['predictions'])) / np.mean(df['labels'])
        data = {
            'predictions': self.predictions,
            'labels': self.labels,
            'groups': self.groups,
            'group_ids': self.group_ids,
        }
        df = pd.DataFrame(data)
        group_mse = df.groupby('group_ids').apply(func)
        tf.logging.info("calc_group_mse {}".format(group_mse.to_string()))
        return group_mse.mean()
    def calc_auc(self):
        def metric(df):
            label = df['labels']
            pred = df['predictions']
            if label.nunique() > 1:
                m = roc_auc_score(label, pred)
            else:
                m = -1
            return m

        data = {
            'predictions': self.predictions,
            'labels': self.labels,
            'groups': self.groups,
        }
        df = pd.DataFrame(data)
        tf.logging.info("calc_auc Get Eval %d data", len(df))
        group_aucs = df.groupby('groups').apply(metric)
        #print(group_aucs)
        avg_auc = group_aucs[group_aucs>=0].mean()  
        return avg_auc
    def calc_seq_auc(self):
        def proc_labels(df):
            return np.max(df["labels"])
        def proc_preds(df):
            return np.sum(df["predictions"]) / 20.
        
        data = {
            'predictions': self.predictions,
            'labels': self.labels,
            'groups': self.groups,
            }
        df = pd.DataFrame(data)
        df_g = df.groupby('groups')
        labels = df_g.apply(proc_labels)
        preds = df_g.apply(proc_preds)
        # print("labels:", labels)
        # print("preds:", preds)
        return roc_auc_score(labels, preds)

class Metric:
    @staticmethod
    def predictions_mean(predictions, weights=None, name=None):
        with ops.name_scope(
                name, 'predictions_mean', (predictions, weights)) as scope:
            predictions = math_ops.to_float(predictions, name='predictions')
            if weights is not None:
                weights = weights_broadcast_ops.broadcast_weights(weights, predictions)
            return metrics_lib.mean(predictions, weights=weights, name=scope)
    
    @staticmethod
    def auc(labels, predictions, weights=None, curve='ROC', name=None):
        with ops.name_scope(name, 'auc', (predictions, labels, weights)) as scope:
            predictions = math_ops.to_float(predictions, name='predictions')
            if labels.dtype.base_dtype != dtypes.bool:
                logging.warning('Casting %s labels to bool.', labels.dtype)
                labels = math_ops.cast(labels, dtypes.bool)
            if weights is not None:
                weights = weights_broadcast_ops.broadcast_weights(weights, predictions)
            return metrics_lib.auc(
                    labels=labels, predictions=predictions, weights=weights, curve=curve,
                    name=scope)
    
    @staticmethod
    def indicator_labels_mean(labels, weights=None, name=None):
        with ops.name_scope(name, 'labels_mean', (labels, weights)) as scope:
            labels = math_ops.to_float(labels, name='labels')
            if weights is not None:
                weights = weights_broadcast_ops.broadcast_weights(weights, labels)
            return metrics_lib.mean(labels, weights=weights, name=scope)
    
    @staticmethod
    def accuracy_baseline(labels_mean):
        with ops.name_scope(None, 'accuracy_baseline', labels_mean):
            value, update_op = labels_mean
            return (
                    math_ops.maximum(value, 1. - value, name='value'),
                    math_ops.maximum(update_op, 1 - update_op, name='update_op'))
    
    @staticmethod
    def accuracy_at_threshold(labels, predictions, weights, threshold, name=None):
        with ops.name_scope(
                name, 'accuracy_at_%s' % threshold,
                (predictions, labels, weights, threshold)) as scope:
            threshold_predictions = math_ops.to_float(
                    math_ops.greater_equal(predictions, threshold))
            return metrics_lib.accuracy(
                    labels=labels, predictions=threshold_predictions, weights=weights,
                    name=scope)
    
    @staticmethod
    def precision_at_threshold(labels, predictions, weights, threshold, name=None):
        with ops.name_scope(
                name, 'precision_at_%s' % threshold,
                (predictions, labels, weights, threshold)) as scope:
            precision_tensor, update_op = metrics_lib.precision_at_thresholds(
                    labels=labels, predictions=predictions, thresholds=(threshold,),
                    weights=weights, name=scope)
            return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)
    
    @staticmethod
    def recall_at_threshold(labels, predictions, weights, threshold, name=None):
        with ops.name_scope(
                name, 'recall_at_%s' % threshold,
                (predictions, labels, weights, threshold)) as scope:
            precision_tensor, update_op = metrics_lib.recall_at_thresholds(
                    labels=labels, predictions=predictions, thresholds=(threshold,),
                    weights=weights, name=scope)
            return array_ops.squeeze(precision_tensor), array_ops.squeeze(update_op)

    @staticmethod
    def auc_cond(pred, label, cond, value, name=""):
        mask = tf.equal(cond, value)
        if label.dtype.base_dtype != dtypes.bool:
            logging.warning('Casting %s label to bool.', label.dtype)
            label = math_ops.cast(label, dtypes.bool)
        auc = tf.contrib.metrics.auc_using_histogram(
                tf.boolean_mask(label, mask), 
                tf.boolean_mask(pred, mask), 
                [0.0, 1.0],
                nbins=1000,
                name=name)
        return auc
    
    @staticmethod
    def metric_variable(shape, dtype, validate_shape=True, name=None):
        return tf.Variable(
            lambda: array_ops.zeros(shape, dtype),
            trainable=False,
            collections=[
                ops.GraphKeys.LOCAL_VARIABLES, 
            ],
            validate_shape=validate_shape,
            name=name)
    
    @staticmethod
    def confusion_matrix_with_decay(labels,
                                    predictions,
                                    thresholds,
                                    weights=None,
                                    includes=None,
                                    decay=1.0):
        # code from tensorflow
        all_includes = ('tp', 'fn', 'tn', 'fp')
        if includes is None:
            includes = all_includes
        else:
            for include in includes:
                if include not in all_includes:
                    raise ValueError('Invalid key: %s.' % include)
    
        with ops.control_dependencies([
                check_ops.assert_greater_equal(
                    predictions,
                    math_ops.cast(0.0, dtype=predictions.dtype),
                    message='predictions must be in [0, 1]'),
                check_ops.assert_less_equal(
                    predictions,
                    math_ops.cast(1.0, dtype=predictions.dtype),
                    message='predictions must be in [0, 1]')
                ]):
            predictions, labels, weights = metric_ops._remove_squeezable_dimensions(
                predictions=math_ops.to_float(predictions),
                labels=math_ops.cast(labels, dtype=dtypes.bool),
                weights=weights)
    
        num_thresholds = len(thresholds)
    
        # Reshape predictions and labels.
        predictions_2d = array_ops.reshape(predictions, [-1, 1])
        labels_2d = array_ops.reshape(
            math_ops.cast(labels, dtype=dtypes.bool), [1, -1])
    
        # Use static shape if known.
        num_predictions = predictions_2d.get_shape().as_list()[0]
    
        # Otherwise use dynamic shape.
        if num_predictions is None:
            num_predictions = array_ops.shape(predictions_2d)[0]
        thresh_tiled = array_ops.tile(
            array_ops.expand_dims(array_ops.constant(thresholds), [1]),
            array_ops.stack([1, num_predictions]))
    
        # Tile the predictions after thresholding them across different thresholds.
        pred_is_pos = math_ops.greater(
            array_ops.tile(array_ops.transpose(predictions_2d), [num_thresholds, 1]),
            thresh_tiled)
        if ('fn' in includes) or ('tn' in includes):
            pred_is_neg = math_ops.logical_not(pred_is_pos)
    
        # Tile labels by number of thresholds
        label_is_pos = array_ops.tile(labels_2d, [num_thresholds, 1])
        if ('fp' in includes) or ('tn' in includes):
            label_is_neg = math_ops.logical_not(label_is_pos)
    
        if weights is not None:
            weights = weights_broadcast_ops.broadcast_weights(
                math_ops.to_float(weights), predictions)
            weights_tiled = array_ops.tile(
                array_ops.reshape(weights, [1, -1]), [num_thresholds, 1])
            thresh_tiled.get_shape().assert_is_compatible_with(
                weights_tiled.get_shape())
        else:
            weights_tiled = None
    
        values = {}
        update_ops = {}
    
        if 'tp' in includes:
            true_p = Metric.metric_variable(
                [num_thresholds], dtypes.float32, name='true_positives')
            is_true_positive = math_ops.to_float(
                math_ops.logical_and(label_is_pos, pred_is_pos))
            if weights_tiled is not None:
                is_true_positive *= weights_tiled
            update_ops['tp'] = state_ops.assign(true_p,
                                                (true_p * decay + 
                                                 math_ops.reduce_sum(
                        is_true_positive, 1)))
            values['tp'] = true_p
    
        if 'fn' in includes:
            false_n = Metric.metric_variable(
                [num_thresholds], dtypes.float32, name='false_negatives')
            is_false_negative = math_ops.to_float(
                math_ops.logical_and(label_is_pos, pred_is_neg))
            if weights_tiled is not None:
                is_false_negative *= weights_tiled
            update_ops['fn'] = state_ops.assign(false_n,
                                                (false_n * decay + 
                                                 math_ops.reduce_sum(
                        is_false_negative, 1)))
            values['fn'] = false_n
    
        if 'tn' in includes:
            true_n = Metric.metric_variable(
                [num_thresholds], dtypes.float32, name='true_negatives')
            is_true_negative = math_ops.to_float(
                math_ops.logical_and(label_is_neg, pred_is_neg))
            if weights_tiled is not None:
                is_true_negative *= weights_tiled
            update_ops['tn'] = state_ops.assign(true_n,
                                                (true_n * decay + 
                                                 math_ops.reduce_sum(
                        is_true_negative, 1)))
            values['tn'] = true_n
    
        if 'fp' in includes:
            false_p = Metric.metric_variable(
                [num_thresholds], dtypes.float32, name='false_positives')
            is_false_positive = math_ops.to_float(
                math_ops.logical_and(label_is_neg, pred_is_pos))
            if weights_tiled is not None:
                is_false_positive *= weights_tiled
            update_ops['fp'] = state_ops.assign(false_p,
                                                (false_p * decay +
                                                 math_ops.reduce_sum(
                        is_false_positive, 1)))
            values['fp'] = false_p
    
        return values, update_ops
    
    @staticmethod
    def area_of_recall(labels,
                       predictions,
                       weights,
                       mask=None,
                       decay=1.0,
                       num_thresholds=300,
                       name=None):
    
        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [
            (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
        ]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
    
        with tf.variable_scope(name, "area_of_recall",
                               values=(labels, predictions, weights, mask)):

            if mask is None:
                mask = 1.
            else:
                mask = tf.where(mask, tf.ones_like(mask, dtype=tf.float32), 
                                tf.zeros_like(mask, dtype=tf.float32))
            
            values_w, update_ops_w = (
                Metric.confusion_matrix_with_decay(
                    labels, predictions, thresholds, 
                    weights=weights*mask, decay=decay))
    
            values, update_ops = (
                Metric.confusion_matrix_with_decay(
                    labels, predictions, thresholds, weights=mask, decay=decay))
    
            def _compute_recall(tp, fn, name):
                epslion = 1e-8
                return tf.divide(tp, tp + fn + epslion,
                                 name="recall_"+name)
    
            def _compute_aor(tp, fn, tn, fp, recall, name):
                epslion = 1e-8
                y = recall
                # sample predicted as positive
                x = ((tp + fp) / 
                     (tp + fn + tn + fp + epslion))
                return tf.reduce_sum(
                    tf.multiply(x[:num_thresholds-1] - x[1:], 
                                (y[:num_thresholds-1] + y[1:]) / 2.0),
                    name=name)
    
            aor_value = _compute_aor(values["tp"], values["fn"],
                                     values["tn"], values["fp"], 
                                     _compute_recall(values_w["tp"], 
                                                     values_w["fn"], 
                                                     name="wighted"),
                                     name="value")
            aor_update = _compute_aor(update_ops["tp"], update_ops["fn"],
                                      update_ops["tn"], update_ops["fp"], 
                                      _compute_recall(update_ops_w["tp"], 
                                                      update_ops_w["fn"],
                                                      name="weighted_up"),
                                      name="update")
            return aor_value, aor_update

    @staticmethod
    def bucketize(values, boundaries, desc=False):
        values= tf.reshape(values, shape=[-1,1])
        boundaries = tf.reshape(boundaries, shape=[1,-1])
        values_tiled = tf.tile(values, [1, boundaries.shape[1]])
        condition = tf.greater_equal(values_tiled, boundaries)
        if desc:
            condition = tf.less_equal(values_tiled, boundaries)
        count = tf.where(condition,
                         tf.ones_like(values_tiled), tf.zeros_like(values_tiled))
        return tf.reduce_sum(count, axis=1)

    @staticmethod
    def frac_recall(labels,
                    predictions,
                    frac=[0.5],
                    get_frac_val=False,
                    mask=None,
                    num_thresholds=300,
                    decay=1.,
                    name=None):
        kepsilon = 1e-7  # to account for floating point imprecisions
        thresholds = [
            (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
        ]
        thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]

        with tf.variable_scope(name, "frac_recall",
                               values=(labels, predictions, frac, mask,)):
            if mask is not None:
                mask = tf.to_float(mask)

            cfm_values, cfm_update_ops = Metric.confusion_matrix_with_decay(
                labels, predictions, thresholds,
                weights=mask, decay=decay)

            def _compute_frac_recall(tp, fn, fp, tn):
                pos_rate = tf.divide(tp+fp, tp+fn+fp+tn+kepsilon)
                index = tf.to_int32(Metric.bucketize(frac, pos_rate, desc=True))
                index = tf.minimum(index, num_thresholds-1)
                if get_frac_val:
                    values = tf.gather(thresholds, index)
                    return values

                recall = tf.divide(tp, tp+fn+kepsilon)
                frac_recall = tf.gather(recall, index)
                return frac_recall

            frac_recall_val = _compute_frac_recall(
                cfm_values['tp'], cfm_values['fn'],
                cfm_values['fp'], cfm_values['tn'])

            frac_recall_ops = _compute_frac_recall(
                cfm_update_ops['tp'], cfm_update_ops['fn'],
                cfm_update_ops['fp'], cfm_update_ops['tn'])

            metrics = {}
            for i in range(len(frac)):
                metrics["frac_%f" % frac[i]] = (frac_recall_val[i],
                                                frac_recall_ops[i])
            return metrics

    @staticmethod
    def seq_pair(labels,
                 predictions,
                 bucket_size,
                 normalize=True,
                 positive=True,
                 weights=None):
        if weights is None:
            weights = tf.ones_like(labels)
        bucket_size = bucket_size + 1
        class _SeqPairMetric:
            def __init__(self):
                self.matrix = np.zeros((bucket_size, bucket_size),
                                       dtype=int)
            def func_update(self, labels, predictions, weights):
                labels = np.reshape(labels, -1)
                predictions = np.reshape(predictions, -1)
                weights = np.reshape(weights, -1)
                fn = lambda x: max(0, min(int(x), bucket_size-1))
                for label, prediction, weight in zip(labels, predictions, weights):
                    label = fn(label)
                    prediction = fn(prediction)
                    self.matrix[prediction][label] += weight
            def func_result(self):
                total = np.sum(self.matrix)
                cnt = 0.
                for i in range(1, bucket_size):
                    for j in range(1, bucket_size):
                        if self.matrix[i][j] == 0:
                            continue
                        if positive:
                            cnt += np.sum(self.matrix[:i, :j]) * self.matrix[i][j]
                        else:
                            cnt += np.sum(self.matrix[i+1:, :j]) * self.matrix[i][j]
                if normalize:
                    return cnt / (total * (total - 1) / 2)
                else:
                    return cnt
        psp = _SeqPairMetric()
        update_op = tf.py_func(psp.func_update, [labels, predictions, weights], [])
        metric_op = tf.py_func(psp.func_result, [], tf.float64)
        return metric_op, update_op


    @staticmethod
    def group_mean(labels, predictions, groups, weights=None):
        if weights == None:
            weights = tf.ones_like(labels)
        gm = _GroupMetric()
        def func_update(labels, predictions, groups, weights):
            labels = np.reshape(labels, -1)
            predictions = np.reshape(predictions, -1)
            weights = np.reshape(weights, -1)
            for label, prediction, wegiht in zip(labels, predictions, weights):
                if wegiht == 0:
                    continue
                for i, group in enumerate(groups):
                    if prediction <= group:
                        gm.add_data(label, prediction, group, i)
                        break
        def func_result():
            return gm.calc_group_mse()
        update_op = tf.py_func(func_update, [labels, predictions, groups, weights], [])
        metric_op = tf.py_func(func_result, [], tf.float64)
        return metric_op, update_op

    @staticmethod
    def group_auc(labels, predictions, groups):
        gm = _GroupMetric()
        def func_update(labels, predictions, groups):
            labels = np.reshape(labels, -1)
            predictions = np.reshape(predictions, -1)
            groups = np.reshape(groups, -1)
            #print ("labels",labels)
            #print ("predictions",predictions)
            #print ("groups",groups)
            gm.add_data(labels, predictions, groups, 0)
        def func_result():
            return gm.calc_auc()
        update_op = tf.py_func(func_update, [labels, predictions, groups], [])
        metric_op = tf.py_func(func_result, [], tf.float64)
        return metric_op, update_op

    @staticmethod
    def seq_auc(labels, predictions, groups):
        gm = _GroupMetric()
        def func_update(labels, predictions, groups):
            labels = np.reshape(labels, -1)
            predictions = np.reshape(predictions, -1)
            groups = np.reshape(groups, -1)
            gm.add_data(labels, predictions, groups, 0)
        def func_result():
            return gm.calc_seq_auc()
        update_op = tf.py_func(func_update, [labels, predictions, groups], [])
        metric_op = tf.py_func(func_result, [], tf.float64)
        return metric_op, update_op        
    
    @staticmethod
    def metric_score(app_mask, apps, scores, wd_scores):
        class _Score:
            def __init__(self):
                self.data = []
            def add_data(self, data):
                self.data.append(data)
            def calc(self):
                if not self.data:
                    return 0.
                else:
                    return sum(self.data) / len(self.data)
        sc = _Score()
        def func_update(app_mask, apps, scores, wd_scores):
            apps = np.reshape(apps, -1)
            scores = np.reshape(scores, -1)
            wd_scores = np.reshape(wd_scores, -1)
            for app, score, wd_score \
                    in zip(apps, scores, wd_scores):
                if app != app_mask or score >= 1 or wd_score >= 1:
                    continue
                if math.fabs(score - wd_score) < 1e-4:
                    sc.add_data(1)
                else:
                    sc.add_data(0)
        def func_result():
            return sc.calc()
        update_op = tf.py_func(func_update, [app_mask, apps, scores, wd_scores], [])
        metric_op = tf.py_func(func_result, [], tf.float64)
        return metric_op, update_op
        
def tets_seq_pair():
    labels = [[1], [2], [2], [3], [4]]
    predictions = [[1], [3], [3], [2], [1]]
    bucket_size = 180
    
    metric_op, update_op = Metric.seq_pair(
            labels, predictions, bucket_size, normalize=False, positive=True)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))
    metric_op, update_op = Metric.seq_pair(
            labels, predictions, bucket_size, normalize=True, positive=True)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))
        
    metric_op, update_op = Metric.seq_pair(
            labels, predictions, bucket_size, normalize=False, positive=False)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))
    metric_op, update_op = Metric.seq_pair(
            labels, predictions, bucket_size, normalize=True, positive=False)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))

def test_bucketize():
    values = tf.reshape([-1.,0.,1.,2.,3.,2.5], shape=[-1,1])
    boundaries = tf.reshape([0.,2.,3.], shape=[-1,1])
    buckets = Metric.bucketize(values, boundaries)
    with tf.Session() as sess:
        print(sess.run(buckets))

def test_frac_recall():
    labels = tf.reshape([1, 1, 0, 0, 1], shape=[-1,1])
    pred = tf.reshape([0.8, 0.6, 0.4, 0.3, 0.2], shape=[-1,1])
    frac = [float(i)/10. for i in range(11)]

    metrics = Metric.frac_recall(labels, pred, frac=frac, get_frac_val=True)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        print(sess.run(metrics))

    metrics = Metric.frac_recall(labels, pred, frac=frac, get_frac_val=False)
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        print(sess.run(metrics))

def test_group_mean():
    labels = [[0.11], [0.22], [0.28], [0.37], [0.45]]
    predictions = [[0.21], [0.32], [0.38], [0.47], [0.55]]
    weights = [[1], [0], [0], [0], [1]]
    groups = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5,0.6]
    metric_op, update_op = Metric.group_mean(
            labels, predictions, groups, weights=weights)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))
    
def test_metric_score():
    scores = tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float32)
    wd_scores = tf.constant([[0.1], [0.3], [0.3]], dtype=tf.float32)
    apps = tf.constant([['browser_iflow'], ['app_iflow'], ['app_iflow']], dtype=tf.string)
    metric_op, update_op = Metric.metric_score(
            'app_iflow', apps, scores, wd_scores)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))
    metric_op, update_op = Metric.metric_score(
            'browser_iflow', apps, scores, wd_scores)
    with tf.Session() as sess:
        print (sess.run([metric_op, update_op]))

def test_group_auc():
    labels = [1,0,0,1,1,1,1,1,0,0]
    #predictions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    predictions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #predictions = [[0.21], [0.32], [0.38], [0.47], [0.55]] * 2
    groups = ["100","101"]*5
    metric_op, update_op = Metric.group_auc(
            labels, predictions, groups)
    with tf.Session() as sess:
        #print (sess.run([update_op,metric_op]))
        print(sess.run([update_op]))
        print(sess.run([metric_op]))

def test_seq_auc():
    labels = [1,0,0,0,1,0,1,0,0,0]
    #predictions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    predictions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #predictions = [[0.21], [0.32], [0.38], [0.47], [0.55]] * 2
    groups = ["100","101"]*5
    metric_op, update_op = Metric.seq_auc(
            labels, predictions, groups)
    with tf.Session() as sess:
        #print (sess.run([update_op,metric_op]))
        print(sess.run([update_op]))
        print(sess.run([metric_op]))


if __name__ == '__main__':
    # tets_seq_pair()
    # test_group_mean()
    # test_group_auc()
    # test_metric_score()
    # test_frac_recall()
    # test_bucketize()
    test_seq_auc()
