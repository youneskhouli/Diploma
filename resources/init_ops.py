from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


@tf_export("keras.initializers.Initializer")
class Initializer(object):

  def __call__(self, shape, dtype=None, partition_info=None):
    raise NotImplementedError

  def get_config(self):
    return {}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf_export("keras.initializers.Zeros", "initializers.zeros",
           "zeros_initializer")
class Zeros(Initializer):

  def __init__(self, dtype=dtypes.float32):
    self.dtype = dtypes.as_dtype(dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return array_ops.zeros(shape, dtype)

  def get_config(self):
    return {"dtype": self.dtype.name}


@tf_export("keras.initializers.VarianceScaling",
           "initializers.variance_scaling", "variance_scaling_initializer")
class VarianceScaling(Initializer):
  def __init__(self,
               scale=1.0,
               mode="fan_in",
               distribution="normal",
               seed=None,
               dtype=dtypes.float32):
    if scale <= 0.:
      raise ValueError("`scale` must be positive float.")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError("Invalid `mode` argument:", mode)
    distribution = distribution.lower()
    if distribution not in {"normal", "uniform"}:
      raise ValueError("Invalid `distribution` argument:", distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution
    self.seed = seed
    self.dtype = _assert_float_dtype(dtypes.as_dtype(dtype))

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal":
      stddev = math.sqrt(scale)
      return random_ops.truncated_normal(
          shape, 0.0, stddev, dtype, seed=self.seed)
    else:
      limit = math.sqrt(3.0 * scale)
      return random_ops.random_uniform(
          shape, -limit, limit, dtype, seed=self.seed)

  def get_config(self):
    return {
        "scale": self.scale,
        "mode": self.mode,
        "distribution": self.distribution,
        "seed": self.seed,
        "dtype": self.dtype.name
    }
