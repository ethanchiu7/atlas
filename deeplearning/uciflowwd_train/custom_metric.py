#coding:utf-8
from __future__ import division 
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


#tf.logging.set_verbosity(tf.logging.INFO)
class GroupMetric(object):

    def __init__(self,name,name_suffix):
        self.pred = []
        self.label = []
        self.group = []
        self._name= name
        self._name_suffix= name_suffix
        self._result = 0.0

    
    def add_data(self, pred, label, group):
        self.pred = np.append(self.pred, pred)
        self.label = np.append(self.label, label)
        self.group = np.append(self.group, group)

    def auc(self):
        def metric(df):
            label = df['label']
            pred = df['pred']
            if label.nunique() > 1:
                m = roc_auc_score(label, pred)
            else:
                m = -1
            return m

        data = {
            'pred': self.pred, 
            'label': self.label, 
            'group': self.group, 
        }
        df = pd.DataFrame(data)
        tf.logging.info("Get Eval %d data", len(df))
        group_aucs = df.groupby('group').apply(metric)
        avg_auc = group_aucs[group_aucs>=0].mean()  
        #print avg_auc
        self._result = avg_auc
    
    def ctr_lift(self):
        def metric(df):
	    #print "===df:",df
            total_num = len(df)
            label = df['label']
	    group = df['group'].values[0]
            click_num =  (label>0.0).sum()
            pred = df['pred']
            ctr = click_num / total_num
            #tf.logging.info("metric %f ctr", ctr)
            mean_pred = pred.mean()
            #group_result = abs(mean_pred/(ctr + 1E-7) - 1)
            #group_result = pow( ctr - mean_pred,2)
            group_ctr_lift = abs( ctr - mean_pred)
            group_ctr_lift = group_ctr_lift / ctr
            #tf.logging.info("metric:%s%s group:%s click_num:%d total_num:%d ctr:%f mean_pred:%f diff:%f lift:%f",self._name,self._name_suffix,group,click_num,total_num, ctr,mean_pred,group_ctr_lift,group_ctr_lift)
            return group_ctr_lift
        data = {
            'pred': self.pred, 
            'label': self.label, 
            'group': self.group, 
        }
        #print "===total_click:",(self.label>0.0).sum()
	#print "===pred_p:",self.pred
        df = pd.DataFrame(data)
        tf.logging.info("Get Eval %d data", len(df))
        group_metric = df.groupby('group').apply(metric)
        #print group_metric
        self._result = group_metric.mean()  

def item_id_score(name,pred, label, group,name_suffix):
    gm = GroupMetric(name,name_suffix)

    def func_update(pred, label, group):
        gm.add_data(pred, label, group)
    

    def func_result():
        gm.ctr_lift()
        return gm._result

    update_op = tf.py_func(func_update, [pred, label, group], [])
    metric_op = tf.py_func(func_result, [], tf.float64)
    return metric_op, update_op
    """
    with tf.Session() as sess: 
        sess.run([tf.global_variables_initializer(),
                     tf.local_variables_initializer()])
        print(sess.run([update_op]))
        print(sess.run([metric_op]))
        #print(sess.run([metric_op,update_op]))
    """

def group_score(name,pred, label,name_suffix, group_config):
    gm = GroupMetric(name,name_suffix)
    gm._group_config = group_config

    def func_update(pred, label):
        #根据配置,计算样本pred所属group
        group = []
        for a_score in pred:
            for a_group in gm._group_config:
                if a_score <= a_group:
                    group.append(a_group)
                    break
        gm.add_data(pred, label, group)
    

    def func_result():
        gm.ctr_lift()
        return gm._result

    update_op = tf.py_func(func_update, [pred, label], [])
    metric_op = tf.py_func(func_result, [], tf.float64)
    return metric_op, update_op
    """
    with tf.Session() as sess: 
        sess.run([tf.global_variables_initializer(),
                     tf.local_variables_initializer()])
        print(sess.run([update_op]))
        print(sess.run([metric_op]))
    """

def group_auc(name,pred, label, group,name_suffix):
    gm = GroupMetric(name,name_suffix)

    def func_update(pred, label, group):
        gm.add_data(pred, label, group)

    def func_result():
        gm.auc()
        return gm._result

    update_op = tf.py_func(func_update, [pred, label, group], [])
    metric_op = tf.py_func(func_result, [], tf.float64)
    return metric_op, update_op
    """
    with tf.Session() as sess: 
        sess.run([tf.global_variables_initializer(),
                     tf.local_variables_initializer()])
        print(sess.run([update_op]))
        print(sess.run([metric_op]))
        #print(sess.run([metric_op,update_op]))
        #print(sess.run([metric_op,update_op]))
    """

def custom_metric(name,pred, label, group=None,name_suffix="",config=None):
    if name == "group_score":
        return group_score(name,pred, label,name_suffix,config)
    elif name == "group_auc":
        return group_auc(name,pred, label, group,name_suffix)
    else: 
        return item_id_score(name,pred, label, group,name_suffix)


if __name__ == "__main__":
    label = [1,1]*5
    pred = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    group = ["100","101"] *5
    custom_metric("item_id_score",pred,label,group,"_base")
    custom_metric("group_score",pred,label,None,"_base",[0.3,0.5,1])
    label = [0,1]*5
    custom_metric("group_auc",pred,label,group,"_base")

