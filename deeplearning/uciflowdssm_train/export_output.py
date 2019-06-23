#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.core.framework import types_pb2
from tensorflow.python.saved_model import utils
from tensorflow.python.estimator.export import export_output
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils

def classification_signature_def(examples, prob):
    if examples is None:
        raise ValueError('RankClassifier examples cannot be None.')
    if not isinstance(examples, ops.Tensor):
        raise ValueError('RankClassifier examples must be a string Tensor.')
    if prob is None:
        raise ValueError('RankClassifier classes and scores cannot both be None.')

    input_tensor_info = utils.build_tensor_info(examples)
    if input_tensor_info.dtype != types_pb2.DT_STRING:
        raise ValueError('RankClassifier examples must be a string Tensor.')
    signature_inputs = {signature_constants.CLASSIFY_INPUTS: input_tensor_info}

    signature_outputs = {}
    if prob is not None:
        scores_tensor_info = utils.build_tensor_info(prob)
        if scores_tensor_info.dtype != types_pb2.DT_FLOAT:
            raise ValueError('RankClassifier scores must be a float Tensor.')
        signature_outputs['prob'] = (scores_tensor_info)

    signature_def = signature_def_utils.build_signature_def(
            signature_inputs, signature_outputs,
            signature_constants.PREDICT_METHOD_NAME)

    return signature_def

class RankClassifierExportOutput(export_output.ExportOutput):
    def __init__(self, prob=None):
        if (prob is not None
            and not (isinstance(prob, ops.Tensor)
                     and prob.dtype.is_floating)):
            raise ValueError('RankClassifierExportOutput scores must be a float32 Tensor; '
                           'got {}'.format(prob))
        if prob is None:
            raise ValueError('prob must be set.')
        self._prob = prob

    @property
    def prob(self):
        return self._prob
    
    def as_signature_def(self, receiver_tensors):
        if len(receiver_tensors) != 1:
            raise ValueError('RankClassifierExportOutput input must be a single string Tensor; '
                             'got {}'.format(receiver_tensors))
        (_, examples), = receiver_tensors.items()
        if dtypes.as_dtype(examples.dtype) != dtypes.string:
            raise ValueError('RankClassifierExportOutput input must be a single string Tensor; '
                         'got {}'.format(receiver_tensors))
        return classification_signature_def(examples, self._prob)
