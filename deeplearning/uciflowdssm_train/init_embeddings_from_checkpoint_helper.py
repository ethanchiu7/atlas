
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.training import session_run_hook
from tensorflow.python.ops import variable_scope

# pylint: disable=protected-access

def _consume_part(name):
    if name.endswith(":0"):
        name = name[:-2]
    if "/part_" in name:
        name = name[:name.index("/part_")]
    return name

def _is_embedding(name):
    name = _consume_part(name)
    #if (name.find("embedding") >= 0):
    #    return False
    return ("dnn/" in name # embeddings defined in EmbeddingColumn
            and (name.endswith("/weights") or name.endswith("/biases")))

def _is_linear(name):
    name = _consume_part(name)
    return ("linear/" in name # embeddings defined in EmbeddingColumn
            and (name.endswith("/weights") or name.endswith("/bias_weight")))

def get_current_deep_variables():
    var_list = tf.get_collection("dnn") # embeddings added in "dnn" collection
    var_map = {}
    for v in var_list:
        var_name = _consume_part(v.name)
        if _is_embedding(var_name):
            var_map[var_name] = v
    return var_map


def get_current_wide_variables():
    var_list = tf.get_collection("linear") # embeddings added in "linear" collection
    var_map = {}
    for v in var_list:
        var_name = _consume_part(v.name)
        var_map[var_name] = v
    return var_map


def create_embedding_map(wide_checkpoint_path, deep_checkpoint_path):
    """ create embedding map from checkpoint_path to current scope
        , for parameter of checkpoint_utils.init_from_checkpoint
    """
    # FOR DEEP SIDE
    # list and find embedding variable
    deep_reader = tf.train.load_checkpoint(deep_checkpoint_path)
    deep_variable_map = deep_reader.get_variable_to_shape_map()
    list_embeddings_checkpoint = []
    for (name,shape) in deep_variable_map.iteritems():
        if _is_embedding(name):
            list_embeddings_checkpoint.append((name,shape))

    # FOR WIDE SIDE
    wide_reader = tf.train.load_checkpoint(wide_checkpoint_path)
    wide_variable_map = wide_reader.get_variable_to_shape_map()
    list_sparse_checkpoint = []
    for (name,shape) in wide_variable_map.iteritems():
        if _is_linear(name):
            list_sparse_checkpoint.append((name,shape))    

    # find in current variable scope
    deep_name_map = {}
    cur_deep_variables = get_current_deep_variables()
    for (name, shape) in list_embeddings_checkpoint:
        if cur_deep_variables.has_key(name):
            print ("-" * 40)
            print ("embedding var name [", name, "], shape = [", shape, "]", cur_deep_variables[name].get_shape())
            deep_name_map[name] = name

    wide_name_map = {}
    cur_wide_variables = get_current_wide_variables()
    for (name, shape) in list_sparse_checkpoint:
        if cur_wide_variables.has_key(name):
            print ("=" * 40)
            print ("sparse var name [", name, "], shape = [", shape, "]", cur_wide_variables[name].get_shape())
            wide_name_map[name] = name

    return wide_name_map, deep_name_map


class InitEmbeddingsHook(session_run_hook.SessionRunHook):
    
    def __init__(self, wide_init_checkpoint, deep_init_checkpoint):
        super(session_run_hook.SessionRunHook, self).__init__()
        self._wide_init_checkpoint = wide_init_checkpoint
        self._deep_init_checkpoint = deep_init_checkpoint

    def begin(self):
        if self._deep_init_checkpoint is None and self._wide_init_checkpoint is None:
            return
        wide_init_map, deep_init_map = create_embedding_map(self._wide_init_checkpoint, self._deep_init_checkpoint)
        print("embeddings to be initialized from {} => {}".format(self._deep_init_checkpoint, deep_init_map))
        print("linear to be initialized from {} => {}".format(self._wide_init_checkpoint, wide_init_map))
        tf.logging.info("embeddings to be initialized from {} => {}".format(self._deep_init_checkpoint, deep_init_map))
        tf.logging.info("linear to be initialized from {} => {}".format(self._wide_init_checkpoint, wide_init_map))
        tf.train.init_from_checkpoint(self._deep_init_checkpoint, deep_init_map)
        tf.train.init_from_checkpoint(self._wide_init_checkpoint, wide_init_map)
        pass

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        pass

    def after_run(self, run_context, run_values):
        pass

    def end(self, session):
        pass
    
# pylint: enable=protected-access
