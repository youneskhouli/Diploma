from __future__ import division

from itertools import chain, combinations
import numbers
import warnings
from itertools import combinations_with_replacement as combinations_w_r
from distutils.version import LooseVersion

import numpy as np
from scipy import sparse
from scipy import stats

from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six import string_types
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.extmath import _incremental_mean_and_var
from ..utils.fixes import _argmax, nanpercentile
from ..utils.sparsefuncs_fast import (inplace_csr_row_normalize_l1,
                                      inplace_csr_row_normalize_l2)
from ..utils.sparsefuncs import (inplace_column_scale,
                                 mean_variance_axis, incr_mean_variance_axis,
                                 min_max_axis)
from ..utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)
from .label import LabelEncoder

BOUNDS_THRESHOLD = 1e-7

zip = six.moves.zip
map = six.moves.map
range = six.moves.range

__all__ = [
    'MinMaxScaler',
    'minmax_scale',
]

def _handle_zeros_in_scale(scale, copy=True):
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        if sparse.issparse(X):
            raise TypeError("MinMaxScaler does no support sparse input. "
                            "You may consider to use MaxAbsScaler instead.")

        X = check_array(X, copy=self.copy, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X):
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X -= self.min_
        X /= self.scale_
        return X


def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
    X = check_array(X, copy=False, ensure_2d=False, warn_on_dtype=True,
                    dtype=FLOAT_DTYPES)
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MinMaxScaler(feature_range=feature_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X