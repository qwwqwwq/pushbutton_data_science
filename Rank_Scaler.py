import warnings
import numbers

import numpy as np
import scipy.sparse as sp
from scipy.stats import scoreatpercentile, percentileofscore

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_arrays
from sklearn.utils import array2d
from sklearn.utils import atleast2d_or_csr
from sklearn.utils import atleast2d_or_csc
from sklearn.utils import safe_asarray
from sklearn.utils import warn_if_not_float
from sklearn.utils.fixes import unique

from sklearn.utils.sparsefuncs import inplace_csr_row_normalize_l1
from sklearn.utils.sparsefuncs import inplace_csr_row_normalize_l2
from sklearn.utils.sparsefuncs import inplace_csr_column_scale
from sklearn.utils.sparsefuncs import mean_variance_axis0


class RankScaler(BaseEstimator, TransformerMixin):

    def __init__(self, copy=True, percision = 2):
        self.copy = copy
	self.percision = percision 

    def fit(self, X, y=None):
        """
	Compute the ranks to be used for later scaling.

	Parameters
	----------
	X : array-like or CSR matrix with shape [n_samples, n_features]
	"""
        X = check_arrays(X, copy=self.copy, sparse_format="csr")[0]
        if sp.issparse(X):
	    #TODO: implement for sparse arrays
	    pass
        else:
	    X.sort()
	    nquantiles = (10**self.percision)
	    scale_factor = float(10**(self.percision-2))
	    self.quantiles = np.array([ scoreatpercentile(X, q/scale_factor) for q in xrange(0,nquantiles) ])
	    self.tform_func = np.vectorize(lambda x: percentileofscore(self.quantiles, x) )
	    self.itform_func = np.vectorize(lambda x: scoreatpercentile(self.quantiles, x) )
            return self

    def transform(self, X, y=None, copy=None):
        """
	Perform standardization by calculating percentile within trained data.

	Parameters
	----------
	X : array-like with shape [n_samples, n_features]
	The data used to scale along the features axis.
	"""
        copy = copy if copy is not None else self.copy
        X = check_arrays(X, copy=copy, sparse_format="csr")[0]
        if sp.issparse(X):
	    #TODO: implement for sparse arrays
	    pass
        else:
	    return (self.tform_func(X)/100)

    def inverse_transform(self, X, copy=None):
        """
	Scale back the data to the original representation

	Parameters
	----------
	X : array-like with shape [n_samples,]
	"""
        if sp.issparse(X):
            if not sp.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
	    X = self.itform_func(X*100.)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
	    X = self.itform_func(X*100.)
        return X
