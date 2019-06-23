import math
import six
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables

Seq = namedtuple('Seq', ['emb', 'seq_len'])

def get_sequence_length(sparse_tensor):
    """
    help to get sequence length of 2-D sparse_tensor.
    """
    batch_size = tf.cast(sparse_tensor.dense_shape[0], dtype=tf.int32)
    sequence_length = (tf.segment_max(
        data=sparse_tensor.indices[:,1]+1,
        segment_ids=sparse_tensor.indices[:,0]))
    extend_zeros = tf.zeros([batch_size - tf.shape(sequence_length)[0]],
                            dtype=tf.int64)
    sequence_length = tf.concat([sequence_length, extend_zeros], axis=0)
    return sequence_length

def process_with_attention(sequence, params):
    """
    turn a sequence to a vector by attention.
    """
    _atten_fn = params.get("atten_fn", None)
    _batch_size = tf.shape(sequence.seq_len)[0]
    _seq_len = tf.cast(tf.reshape(sequence.seq_len, shape=[-1,1]), dtype=tf.int32)
    _max_step = tf.cast(tf.reduce_max(sequence.seq_len), dtype=tf.int32)
    _state_size = sequence.emb.shape[2]
    target_values = params["target_values"]

    _cond = lambda time, *_: time < _max_step
    def _step(time, acc_values, acc_weights):
        values = tf.reshape(sequence.emb[:,time,:], shape=[-1,_state_size])
        cor_ratio = tf.reshape(_atten_fn(values, target_values), shape=[-1,1])
        cor_ratio = tf.where(time < _seq_len,
                             cor_ratio, tf.zeros_like(cor_ratio))
        
        weighted_values = values * cor_ratio
        return time+1, acc_values+weighted_values, acc_weights+cor_ratio

    time = tf.constant(0, dtype=tf.int32, name="time")
    acc_values = tf.zeros(shape=[_batch_size,_state_size],
                          dtype=tf.float32, name="acc_values")
    acc_weights = tf.zeros(shape=[_batch_size,1],
                           dtype=tf.float32, name="acc_weights")
    f_time, f_values, f_weights = tf.while_loop(
        cond=_cond, body=_step,
        loop_vars=[time, acc_values, acc_weights])

    if params.get("normalize_weights", True):
        weighted_values = f_values / (f_weights + 1e-7)
    else:
        weighted_values = f_values
    return None, weighted_values

def sequence_to_vector(sequence, params):
    """
    turn a sequence to a vector.

    inputs: 
    1) sequence: 
      namedtuple: Seq(emb, seq_len),
      rank=2, like SparseTensor
        i-row for i-th samples, 
        j-col for j-th sequence element of i-th samples

    output:
    1) outputs:
      a 3-D tensor: [batch_size, max_time, state_size]
    2) state:
      a 2-D tensor: [batch_size, state_size]
    """

    _root_scope = params.get("name_scope", "seq_to_vec")
    with tf.name_scope(None, _root_scope,
                       values=([a for a in sequence]
                               +list(six.itervalues(params)))):
        return process_with_attention(sequence, params)

class SeqModelSimple:
    def __init__(self, features, mode, params):
        self._root_scope = params.get("seq_model_scope", "seq")
        self._features = features
        self._mode = mode
        self._params = params
        self._num_ps_replicas = int(params.get("num_ps_replicas"))
        self._itemId_hash_bucket_size = int(params.get("itemId_hash_bucket_size", 1e7))
        self._itemId_embedding_size = int(params.get("itemId_embedding_size", 32))
        self._hidden_units = params.get("hidden_units", [32, 16])

    def calc_vectors(self):
        with tf.variable_scope("calc_vectors",
                               values=tuple(six.itervalues(self._features))):
            # in this simple case, create ClkI x itemId seq model
            ClkI_sparse = self._features["ClkI"]
            ClkI_seq_len = get_sequence_length(ClkI_sparse)
            itemId_dense = self._features["item_id"]
            itemId_dense = tf.reshape(itemId_dense, shape=[-1])
            print("itemId_dense: ", itemId_dense)

            # step-1, hash raw features
            itemId_hash = tf.string_to_hash_bucket_fast(
                itemId_dense,
                self._itemId_hash_bucket_size)
            ClkI_val_hash = tf.string_to_hash_bucket_fast(
                ClkI_sparse.values,
                self._itemId_hash_bucket_size)
            print("itemId_hash: ", itemId_hash)
            print("ClkI_val_hash: ", ClkI_val_hash)

            # step-1.1, sparse_to_dense
            ClkI_hash = tf.sparse_to_dense(sparse_indices=ClkI_sparse.indices,
                                           output_shape=ClkI_sparse.dense_shape,
                                           sparse_values=ClkI_val_hash)

            # step-2, embedding lookup
            with tf.variable_scope("embedding_tables", reuse=False):
                _itemId_emb_shape = [self._itemId_hash_bucket_size,
                                     self._itemId_embedding_size]
                self._itemId_emb_tab = tf.contrib.framework.model_variable(
                    name="item_id_embedding/weights",
                    shape=_itemId_emb_shape,
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(
                        mean=0.,
                        stddev=1./math.sqrt(self._itemId_hash_bucket_size)),
                    trainable=True,
                    collections=[self._root_scope],
                    partitioner=tf.fixed_size_partitioner(10, axis=0))

            with tf.name_scope("embedding_lookup",
                               values=(self._itemId_emb_tab, itemId_hash, ClkI_hash)):
                itemId_emb = tf.nn.embedding_lookup(self._itemId_emb_tab, itemId_hash)
                ClkI_emb = tf.nn.embedding_lookup(self._itemId_emb_tab, ClkI_hash)
                self._itemId_emb = itemId_emb

            # step-3, proccess sequence_to_vector
            def _atten_fn_mlp(a, b):
                _hidden_units = self._hidden_units
                with tf.variable_scope("atten_fn_linear",
                                       reuse=tf.AUTO_REUSE,
                                       values=(a,b)) as atten_fn_scope:
                    size_a = a.shape[1]
                    size_b = b.shape[1]
                    c = tf.matmul(
                        tf.reshape(a, shape=[-1, size_a, 1]),
                        tf.reshape(b, shape=[-1, 1, size_b]),
                        name="similarity")
                    c = tf.reshape(c, shape=[-1, size_a * size_b])
                    net = tf.concat([a,b,c], axis=1)
                    for layer_id, num_units in enumerate(_hidden_units):
                        with tf.variable_scope(
                                "hidden_layer_%d" % layer_id,
                                values=(net,)) as hidden_layer_scope:
                           net = tf.contrib.layers.fully_connected(
                               net, num_units,
                               activation_fn=tf.nn.relu,
                               variables_collections=[self._root_scope],
                               scope=hidden_layer_scope)
                    _output = tf.contrib.layers.fully_connected(
                        net, 1,
                        activation_fn=tf.exp,
                        variables_collections=[self._root_scope],
                        scope=atten_fn_scope)
                    return _output
            
            atten_params = {
                "proc_type": "atten",
                "target_values": self._itemId_emb,
                "atten_fn": _atten_fn_mlp,
                "normalize_weights": False,
            }
            outputs, ClkI_vec = sequence_to_vector(
                Seq(emb=ClkI_emb, seq_len=ClkI_seq_len), atten_params)
            self._clki_vec = ClkI_vec
            self._vectors = {
                "seq2vec_ClkI_vec": self._clki_vec,
                "seq2vec_itemId_vec": self._itemId_emb,
                }
            return self._vectors
