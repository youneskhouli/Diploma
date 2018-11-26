from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.platform import tf_logging as logging
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Aliases for some automatically-generated names.
linspace = gen_math_ops.lin_space

arg_max = deprecation.deprecated(None, "Use `argmax` instead")(arg_max)  # pylint: disable=used-before-assignment
arg_min = deprecation.deprecated(None, "Use `argmin` instead")(arg_min)  # pylint: disable=used-before-assignment
tf_export("arg_max")(arg_max)
tf_export("arg_min")(arg_min)

# This is set by resource_variable_ops.py. It is included in this way since
# there is a circular dependency between math_ops and resource_variable_ops
_resource_variable_type = None


def _set_doc(doc):

  def _decorator(func):
    func.__doc__ = doc
    return func

  return _decorator

@tf_export("reduce_mean")
@deprecation.deprecated_args(
    None, "keep_dims is deprecated, use keepdims instead", "keep_dims")
def reduce_mean(input_tensor,
                axis=None,
                keepdims=None,
                name=None,
                reduction_indices=None,
                keep_dims=None):

  keepdims = deprecation.deprecated_argument_lookup("keepdims", keepdims,
                                                    "keep_dims", keep_dims)

  if keepdims is None:
    keepdims = False
  return _may_reduce_to_scalar(keepdims, axis, reduction_indices,
                               gen_math_ops.mean(
                                   input_tensor,
                                   _ReductionDims(input_tensor, axis,
                                                  reduction_indices),
                                   keepdims,
                                   name=name))