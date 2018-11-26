from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
# pylint: enable=wildcard-import

# Used for slicing to specify a new 1 size dimension
newaxis = None
tf_export("newaxis").export_constant(__name__, "newaxis")

# We override the 'slice' for the "slice" op, so we keep python's
# existing 'slice' for later use in this module.
_BaseSlice = slice


@tf_export("identity")
def identity(input, name=None):
  if context.executing_eagerly():
    input = ops.convert_to_tensor(input)
    in_device = input.device
    context_device = context.context().device_name
    if not context_device:
      context_device = "/job:localhost/replica:0/task:0/device:CPU:0"
    if context_device != in_device:
      return input._copy()
    return input
  else:
    return gen_array_ops.identity(input, name=name)


@tf_export("expand_dims")
@deprecation.deprecated_args(None, "Use the `axis` argument instead", "dim")
def expand_dims(input, axis=None, name=None, dim=None):
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dim", dim)
  return gen_array_ops.expand_dims(input, axis, name)

listdiff.__doc__ = gen_array_ops.list_diff.__doc__ + "\n" + listdiff.__doc__

@tf_export("setdiff1d")
def setdiff1d(x, y, index_dtype=dtypes.int32, name=None):
  return gen_array_ops.list_diff(x, y, index_dtype, name)

setdiff1d.__doc__ = gen_array_ops.list_diff.__doc__

@tf_export("shape")
def shape(input, name=None, out_type=dtypes.int32):
  return shape_internal(input, name, optimize=True, out_type=out_type)

def shape_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  with ops.name_scope(name, "Shape", [input]) as name:
    if isinstance(input, (sparse_tensor.SparseTensor,
                          sparse_tensor.SparseTensorValue)):
      return gen_math_ops.cast(input.dense_shape, out_type)
    else:
      if not context.executing_eagerly():
        input_tensor = ops.convert_to_tensor(input)
        input_shape = input_tensor.get_shape()
        if optimize and input_shape.is_fully_defined():
          return constant(input_shape.as_list(), out_type, name=name)
      return gen_array_ops.shape(input, name=name, out_type=out_type)

@tf_export("size")
def size(input, name=None, out_type=dtypes.int32):
  return size_internal(input, name, optimize=True, out_type=out_type)

def size_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  if context.executing_eagerly() and not isinstance(
      input, (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    input = ops.convert_to_tensor(input)
    np_out_type = out_type.as_numpy_dtype
    num_elements = np.prod(input._shape_tuple(), dtype=np_out_type)
    return ops.convert_to_tensor(num_elements, dtype=out_type)
  with ops.name_scope(name, "Size", [input]) as name:
    if isinstance(input, (sparse_tensor.SparseTensor,
                          sparse_tensor.SparseTensorValue)):
      return gen_math_ops.prod(
          gen_math_ops.cast(input.dense_shape, out_type), 0, name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize:
        if input_shape.is_fully_defined():
          return constant(input_shape.num_elements(), out_type, name=name)
        if input_shape.dims and any(dim == 0 for dim in input_shape.dims):
          return constant(0, out_type, name=name)
      return gen_array_ops.size(input, name=name, out_type=out_type)

@tf_export("transpose")
def transpose(a, perm=None, name="transpose", conjugate=False):
  with ops.name_scope(name, "transpose", [a]) as name:
    transpose_fn = (
        gen_array_ops.conjugate_transpose
        if (conjugate and a.dtype.is_complex) else gen_array_ops.transpose)
    if perm is None:
      rank = gen_array_ops.rank(a)
      perm = (rank - 1) - gen_math_ops._range(0, rank, 1)
      ret = transpose_fn(a, perm, name=name)
      if not context.executing_eagerly():
        input_shape = ret.op.inputs[0].get_shape().dims
        if input_shape is not None:
          ret.set_shape(input_shape[::-1])
    else:
      ret = transpose_fn(a, perm, name=name)
    return ret

@tf_export("placeholder")
def placeholder(dtype, shape=None, name=None):
  if context.executing_eagerly():
    raise RuntimeError("tf.placeholder() is not compatible with "
                       "eager execution.")

  return gen_array_ops.placeholder(dtype=dtype, shape=shape, name=name)

def _normalize_sparse_shape(shape, name):
  if shape is None:
    return (None, None)
  rank = shape.get_shape()[0] if isinstance(shape, ops.Tensor) else len(shape)
  if not isinstance(shape, ops.Tensor) and None in shape:
    return (None, rank)
  return (ops.convert_to_tensor(shape, dtype=dtypes.int64, name=name), rank)


@tf_export("sparse_placeholder")
def sparse_placeholder(dtype, shape=None, name=None):
  if context.executing_eagerly():
    raise RuntimeError("tf.placeholder() is not compatible with "
                       "eager execution.")

  shape_name = (name + "/shape") if name is not None else None
  shape, rank = _normalize_sparse_shape(shape, shape_name)
  if shape is None:
    shape = placeholder(dtypes.int64, shape=[rank], name=shape_name)
  return sparse_tensor.SparseTensor(
      values=placeholder(
          dtype,
          shape=[None],
          name=(name + "/values") if name is not None else None),
      indices=placeholder(
          dtypes.int64, shape=[None, rank],
          name=(name + "/indices") if name is not None else None),
      dense_shape=shape)