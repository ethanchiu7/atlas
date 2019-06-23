
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

def get_current_variables():
    var_list = tf.get_collection("dnn") + tf.get_collection("linear") + tf.get_collection("seq")
    print("****************var_list****************")
    for var in var_list:
        print(var)

    var_map = {}
    for v in sorted(var_list, key=lambda v: v.name):
        var_name = _consume_part(v.name)
        print('{0}->{1}->{2}'.format(var_name, v.name, v.shape))
        if not var_name in var_map:
            shape = v.shape
        else:
            print("var_name: ", var_name)
            a1, b1 = var_map[var_name]
            a2, b2 = v.shape
            assert(b1 == b2)
            shape = [a1+a2, b1]
        var_map[var_name] = shape
    return var_map

def init_self_name_map(name_map, opt_var_list):
    for name in opt_var_list:
        print("**********name**********")
        print(name)
        name_map[name] = name
    return name_map

def create_embedding_map(checkpoint_path):
    """ create embedding map from checkpoint_path to current scope
        , for parameter of checkpoint_utils.init_from_checkpoint
    """
    # list and find embedding variable
    reader = checkpoint_utils.load_checkpoint(checkpoint_path)
    variable_map = reader.get_variable_to_shape_map()
    list_embeddings_checkpoint = []
    opt_var_list = []
    for (name, shape) in sorted(variable_map.items(), key=lambda (name, shape): name):
        list_embeddings_checkpoint.append((name, shape))
    print("****************list_embeddings_checkpoint**************")
    print(list_embeddings_checkpoint)

    # find in current variable scope
    name_map = {}
    cur_variables = get_current_variables()
    print("****************cur_var*************")
    print(cur_variables)
    print("****************info****************")
    for (name, shape) in list_embeddings_checkpoint:
        print("init_model: {0}->{1}".format(name, shape))
        if cur_variables.has_key(name):
            print("find {0} in current model".format(name))
            if shape == cur_variables[name]:
                name_map[name] = name
                print("shape no-diff, add {0} to name_map".format(name))
            else:
                print("shape diff [{0} v.s {1}], ignore {2}".format(shape, cur_variables[name], name))
        else:
            print("can not find {0} in current model".format(name))
    print("****************name_map**********************")
    for key, val in name_map.items():
        print('{0}->{1}'.format(key, val))
    # print(name_map)
    return name_map

class InitEmbeddingsHook(session_run_hook.SessionRunHook):
    
    def __init__(self, init_checkpoint):
        super(session_run_hook.SessionRunHook, self).__init__()
        self._init_checkpoint = init_checkpoint

    def begin(self):
        if self._init_checkpoint is None: return
        init_map = create_embedding_map(self._init_checkpoint)
        tf.logging.info("embeddings to be initialized from {}: {}".format(self._init_checkpoint, init_map))
        checkpoint_utils.init_from_checkpoint(self._init_checkpoint, init_map)
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
