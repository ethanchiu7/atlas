#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import logging
import numpy as np
import tensorflow as tf
from abc import ABCMeta
from model_fn import self_model_fn

logger = logging.getLogger()

class ModelBuilderError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)

class ParametersError(ModelBuilderError):
    def __init__(self, msg):
        super(ParametersError, self).__init__(msg)

class TensorTransform(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters

    def get_value_tf_type(self, type_name):
        type_value = self.parameters.get(type_name)
        if type_value == "float32":
            return tf.float32
        elif type_value == "float64":
            return tf.float64
        elif type_value == "int8":
            return tf.int8
        elif type_value == "int16":
            return tf.int16
        elif type_value == "int64":
            return tf.int64
        else:
            return None

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        input_tensor = output_tensors.get(input_tensor_name)
        output_tensor_name = self.parameters.get("output_tensor")
        output_tensors[output_tensor_name] = input_tensor

    
class SparseColumnWithHashBucket(TensorTransform):
    def __init__(self, name, parameters):
        super(SparseColumnWithHashBucket, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        dtype = self.get_value_tf_type("dtype")
        if dtype == None:
            dtype = tf.string
        if self.parameters.has_key("hash_bucket_size"):
            hash_bucket_size = self.parameters.get("hash_bucket_size")
        else:
            msg = "parameters error, sparse_column_with_hash_bucket must need hash_bucket_size"
            logger.error(msg)
            raise ParametersError(msg)
        if self.parameters.has_key("combiner"):
            combiner = self.parameters.get("combiner")
            output_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(
                column_name=input_tensor_name,
                hash_bucket_size=hash_bucket_size,
                combiner=combiner,
                dtype=dtype)
        else:
            output_tensor = tf.contrib.layers.sparse_column_with_hash_bucket(
                column_name=input_tensor_name,
                hash_bucket_size=hash_bucket_size,
                dtype=dtype)
        output_tensors[output_tensor_name] = output_tensor

class RealValuedColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(RealValuedColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        dtype = self.get_value_tf_type("dtype")
        if dtype == None:
            dtype = tf.float32
        output_tensor = tf.contrib.layers.real_valued_column(
            input_tensor_name, dtype=dtype)
        output_tensors[output_tensor_name] = output_tensor

class EmbeddingColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(EmbeddingColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("dimension"):
            dimension = self.parameters.get("dimension")
        else:
            msg = "parameters error, embedding_column must need dimension"
            logger.error(msg)
            raise ParametersError(msg)
        input_tensor = output_tensors.get(input_tensor_name)
        ckpt_to_load_from = None
        tensor_name_in_ckpt = None
        if self.parameters.has_key("ckpt_to_load_from") and self.parameters.has_key("tensor_name_in_ckpt"):
            ckpt_to_load_from = self.parameters.get("ckpt_to_load_from")
            tensor_name_in_ckpt = self.parameters.get("tensor_name_in_ckpt")
        if self.parameters.has_key("combiner"):
            combiner = self.parameters.get("combiner")
            output_tensor = tf.contrib.layers.embedding_column(
                sparse_id_column=input_tensor,
                dimension=dimension,
                combiner=combiner,
                ckpt_to_load_from=ckpt_to_load_from,
                tensor_name_in_ckpt=tensor_name_in_ckpt)
        else:
            output_tensor = tf.contrib.layers.embedding_column(
                sparse_id_column=input_tensor, 
                dimension=dimension,
                ckpt_to_load_from=ckpt_to_load_from,
                tensor_name_in_ckpt=tensor_name_in_ckpt)
        output_tensors[output_tensor_name] = output_tensor


class SparseColumnWithKeys(TensorTransform):
    def __init__(self, name, parameters):
        super(SparseColumnWithKeys, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("keys"):
            keys = self.parameters.get("keys")
        else:
            msg = "parameters error, sparse_column_with_keys must need keys"
            logger.error(msg)
            raise ParametersError(msg)
        default_value = self.parameters.get("default_value", -1)
        combiner = self.parameters.get("combiner", 'sum')
        dtype = self.get_value_tf_type("dtype")
        if dtype == None:
            dtype = tf.string
        output_tensors[output_tensor_name] = tf.contrib.layers.sparse_column_with_keys(
            column_name=input_tensor_name,
            keys=keys,
            default_value=default_value,
            combiner=combiner,
            dtype=dtype)

class SparseColumnWithIntegerizedFeature(TensorTransform):
    def __init__(self, name, parameters):
        super(SparseColumnWithIntegerizedFeature, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        combiner = self.parameters.get("combiner", 'sum')
        if self.parameters.has_key("bucket_size"):
            bucket_size = self.parameters.get("bucket_size")
        else:
            msg = "parameters error, sparse_column_with_integerized_feature must need bucket_size"
            logger.error(msg)
            raise ParametersError(msg)
        dtype = self.get_value_tf_type("dtype")
        if dtype == None:
            dtype = tf.int64
        output_tensors[output_tensor_name] = tf.contrib.layers.sparse_column_with_integerized_feature(
            column_name=input_tensor_name,
            bucket_size=bucket_size,
            combiner=combiner,
            dtype=dtype)

class BucketizedColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(BucketizedColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("boundaries"):
            boundaries = self.parameters.get("boundaries")
            if not isinstance(boundaries, list):
                boundaries = str(boundaries).replace(' ', '')
                pattern = re.compile('np.linspace\(([0-9]+\.[0-9]+),([0-9]+\.[0-9]+),([0-9]+\.[0-9]+)\)')
                result = pattern.findall(boundaries)
                boundaries = list(np.linspace(float(result[0][0]),
                                              float(result[0][1]),
                                              float(result[0][2])))
        else:
            msg = "parameters error, sparse_column_with_keys must need keys"
            logger.error(msg)
            raise ParametersError(msg)
        input_tensor = output_tensors.get(input_tensor_name)
        output_tensors[output_tensor_name] = tf.contrib.layers.bucketized_column(
            source_column=input_tensor,
            boundaries=boundaries)


class SharedEmbeddingColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(SharedEmbeddingColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor = list()
        for input_tensor_name in self.parameters.get("input_tensor"):
            input_tensor.append(output_tensors.get(input_tensor_name))
        if self.parameters.has_key("dimension"):
            dimension = self.parameters.get("dimension")
        else:
            msg = "parameters error, embedding_column must need dimension"
            logger.error(msg)
            raise ParametersError(msg)
        ckpt_to_load_from = None
        tensor_name_in_ckpt = None
        if self.parameters.has_key("ckpt_to_load_from") and self.parameters.has_key("tensor_name_in_ckpt"):
            ckpt_to_load_from = self.parameters.get("ckpt_to_load_from")
            tensor_name_in_ckpt = self.parameters.get("tensor_name_in_ckpt")
        if self.parameters.has_key("combiner"):
            combiner = self.parameters.get("combiner")
            shared_embedding_columns = tf.contrib.layers.shared_embedding_columns(sparse_id_columns=input_tensor,
                                                               dimension=dimension,
                                                               combiner=combiner,
                                                               ckpt_to_load_from=ckpt_to_load_from,
                                                               tensor_name_in_ckpt=tensor_name_in_ckpt)
        else:
            shared_embedding_columns = tf.contrib.layers.shared_embedding_columns(sparse_id_columns=input_tensor,
                                                               dimension=dimension,
                                                               ckpt_to_load_from=ckpt_to_load_from,
                                                               tensor_name_in_ckpt=tensor_name_in_ckpt)

        for output_tensor_name, output_tensor in zip(self.parameters.get("output_tensor"), shared_embedding_columns):
            output_tensors[output_tensor_name] = output_tensor


class CrossedColumn(TensorTransform):
    def __init__(self, name, parameters):
        super(CrossedColumn, self).__init__(name, parameters)

    def transform(self, output_tensors):
        input_tensor_name = self.parameters.get("input_tensor")
        output_tensor_name = self.parameters.get("output_tensor")
        if self.parameters.has_key("hash_bucket_size"):
            hash_bucket_size = self.parameters.get("hash_bucket_size")
        else:
            msg = "parameters error, crossed_column must need hash_bucket_size"
            logger.error(msg)
            raise ParametersError(msg)
        column_names = input_tensor_name.split(",")
        columns = []
        for index in range(len(column_names)):
            input_tensor = output_tensors.get(column_names[index])
            columns.append(input_tensor)
        if self.parameters.has_key("combiner"):
            combiner = self.parameters.get("combiner")
            output_tensor = tf.contrib.layers.crossed_column(
                columns=columns,
                hash_bucket_size=hash_bucket_size,
                combiner=combiner)
        else:
            output_tensor = tf.contrib.layers.crossed_column(
                columns=columns,
                hash_bucket_size=hash_bucket_size)
        output_tensors[output_tensor_name] = output_tensor


class WideNDeepModelBuilder(object):
    def __init__(self, model_desc=None):
        self.model_desc = model_desc
        self.output_tensors = dict()

    def process_tensor_transform(self, name, parameters):
        if name == "sparse_column_with_hash_bucket":
            tensor_transform = SparseColumnWithHashBucket(name, parameters)
        elif name == "real_valued_column":
            tensor_transform = RealValuedColumn(name, parameters)
        elif name == "embedding_column":
            tensor_transform = EmbeddingColumn(name, parameters)
        elif name == "shared_embedding_column":
            tensor_transform = SharedEmbeddingColumn(name, parameters)
        elif name == "crossed_column":
            tensor_transform = CrossedColumn(name, parameters)
        elif name == "sparse_column_with_keys":
            tensor_transform = SparseColumnWithKeys(name, parameters)
        elif name == "sparse_column_with_integerized_feature":
            tensor_transform = SparseColumnWithIntegerizedFeature(name, parameters)
        elif name == "bucketized_column":
            tensor_transform = BucketizedColumn(name, parameters)
        elif name == "":
            tensor_transform = TensorTransform(name, parameters)
        else:
            msg = "transform %s is error or not supported" % name
            logger.error(msg)
            raise ParametersError(msg)
        tensor_transform.transform(self.output_tensors)
    
    def build_estimator(self, model_fn_param):
        wide = []
        deep = []
        for tensor_transform in self.model_desc.get("tensorTransform"):
            name = tensor_transform.get("name")
            parameters = tensor_transform.get("parameters")
            print ("Process transform %s, input tensor: %s, "
                  "output tensor: %s, wide or deep select: %s"
                  % (name, parameters.get("input_tensor"),
                     parameters.get("output_tensor"),
                     parameters.get("wide_or_deep")))
            self.process_tensor_transform(name, parameters)
            wide_or_deep = parameters.get("wide_or_deep")
            if type(parameters.get("output_tensor")) != list:
                parameters["output_tensor"] = [parameters["output_tensor"]]
            for output_tensor_name in parameters.get("output_tensor"):
                out_tensor = self.output_tensors.get(output_tensor_name)
                if not out_tensor:
                    msg = "transform %s process error, output tensor is null" % name
                    logger.error(msg)
                    raise ParametersError(msg)
                if wide_or_deep == "wide":
                    wide.append(out_tensor)
                elif wide_or_deep == "deep":
                    deep.append(out_tensor)
                else:
                    print ("column %s not used in wide or deep" % output_tensor_name)
        model_config = self.model_desc.get("modelConfig")
        model_type = model_config.get("model_type")
        parameters = model_config.get("parameters")
        checkpoint_path = parameters.get("checkpoint_path")
        dnn_hidden_units = parameters.get('dnn_hidden_units')
        
        print ("*******************wide columns*******************")
        for i in range(len(wide)):
            print (wide[i])
        print ("*******************deep columns*******************")
        for i in range(len(deep)):
            print (deep[i])
        
        run_config = model_fn_param.get('run_config')
        if model_type == "wide_n_deep":
            m = tf.estimator.DNNLinearCombinedClassifier(
                model_dir=checkpoint_path,
                linear_feature_columns=wide,
                dnn_feature_columns=deep,
                dnn_hidden_units=dnn_hidden_units,
                weight_column=model_fn_param.get('weight_column'),
                config=model_fn_param.get('run_config'))
        elif model_type == "self_define":
            params = {
                'wide': wide,
                'deep': deep,
                'dnn_hidden_units': dnn_hidden_units,
                'num_ps_replicas': run_config.num_ps_replicas if run_config else 0,
                'model_type': parameters.get('model_type'),
            }
            params.update(model_fn_param)
            print ('============== model_fn_param start ===============')
            for k, v in params.items():
                print (k, v)
            print ('============== model_fn_param end ===============')
            m = tf.estimator.Estimator(model_fn=self_model_fn,
                                       model_dir=checkpoint_path,
                                       config=run_config,
                                       params=params)
        return m
