# -*- coding: utf-8 -*-

"""
"""

from __future__ import print_function, division

import numpy as np
import warnings

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from . import wrapper


def _fit_gensvm(X, y, p, lmd, kappa, epsilon, weight_idx, kernel, gamma, coef, 
        degree, kernel_eigen_cutoff, verbose, max_iter, random_state=None):

    # process the random state
    rnd = check_random_state(random_state)

    # set the verbosity in GenSVM
    wrapper.set_verbosity_wrap(verbose)

    # run the actual training
    raw_coef_, n_SV_, n_iter_, training_error_, status_ = wrapper.train_wrap(
            X, y, p, lmd, kappa, epsilon, weight_idx, kernel, gamma, coef, 
            degree, kernel_eigen_cutoff, max_iter, 
            rnd.randint(np.iinfo('i').max))

    # process output
    if status_ == 1 and verbose > 0:
        warnings.warn("GenSVM optimization prematurely ended due to a "
                "incorrect step in the optimization algorithm.", 
                FitFailedWarning)

    if status_ == 2 and verbose > 0:
        warnings.warn("GenSVM failed to converge, increase "
                "the number of iterations.", ConvergenceWarning)

    coef_ = raw_coef_[1:, :]
    intercept_ = raw_coef_[0, :]

    return coef_, intercept_, n_iter_, n_SV_


class GenSVM(BaseEstimator):
    """Generalized Multiclass Support Vector Machine Classification.

    This class implements the basic GenSVM classifier. GenSVM is a generalized 
    multiclass SVM which is flexible in the weighting of misclassification 
    errors. It is this flexibility that makes it perform well on diverse 
    datasets.

    This methods of this class use the GenSVM C library for the actual 
    computations.

    Parameters
    ----------
    p : float, optional (default=1.0)
        Parameter for the L_p norm of the loss function (1.0 <= p <= 2.0)

    lmd : float, optional (default=1e-5)
        Parameter for the regularization term of the loss function (lmd > 0)

    kappa : float, optional (default=0.0)
        Parameter for the hinge function in the loss function (kappa > -1.0)

    weight_idx : int, optional (default=1)
        Type of sample weights to use (1 = unit weights, 2 = size correction 
        weights)

    kernel : string, optional (default='linear')
        Specify the kernel type to use in the classifier. It must be one of 
        'linear', 'poly', 'rbf', or 'sigmoid'.

    gamma : float, optional (default=1.0)
        Kernel parameter for the rbf, poly, and sigmoid kernel

    coef : float, optional (default=0.0)
        Kernel parameter for the poly and sigmoid kernel

    degree : float, optional (default=2.0)
        Kernel parameter for the poly kernel

    kernel_eigen_cutoff : float, optional (default=1e-8)
        Cutoff point for the reduced eigendecomposition used with 
        kernel-GenSVM. Eigenvectors for which the ratio between their 
        corresponding eigenvalue and the largest eigenvalue is smaller than the 
        cutoff will be dropped.

    verbose : int, (default=0)
        Enable verbose output

    max_iter : int, (default=1e8)
        The maximum number of iterations to be run.


    Attributes
    ----------
    coef_ : array, shape = [n_features, n_classes-1]
        Weights assigned to the features (coefficients in the primal problem)

    intercept_ : array, shape = [n_classes]
        Constants in the decision function

    n_iter_ : int
        The number of iterations that were run during training.

    n_support_ : int
        The number of support vectors that were found


    References
    ----------
    * Van den Burg, G.J.J. and Groenen, P.J.F.. GenSVM: A Generalized 
    Multiclass Support Vector Machine. Journal of Machine Learning Research, 
    17(225):1--42, 2016.

    """

    def __init__(self, p=1.0, lmd=1e-5, kappa=0.0, epsilon=1e-6, weight_idx=1, 
            kernel='linear', gamma=1.0, coef=0.0, degree=2.0, 
            kernel_eigen_cutoff=1e-8, verbose=0, random_state=None, 
            max_iter=1e8):
        self.p = p
        self.lmd = lmd
        self.kappa = kappa
        self.epsilon = epsilon
        self.weight_idx = weight_idx
        self.kernel = kernel
        self.gamma = gamma
        self.coef = coef
        self.degree = degree
        self.kernel_eigen_cutoff = kernel_eigen_cutoff
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter


    def fit(self, X, y):
        if not 1.0 <= self.p <= 2.0:
            raise ValueError("Value for p should be within [1, 2]; got p = %r)" 
                    % self.p)
        if not self.kappa > -1.0:
            raise ValueError("Value for kappa should be larger than -1; got "
                    "kappa = %r" % self.kappa)
        if not self.lmd > 0:
            raise ValueError("Value for lmd should be larger than 0; got "
                    "lmd = %r" % self.lmd)
        if not self.epsilon > 0:
            raise ValueError("Value for epsilon should be larger than 0; got "
                    "epsilon = %r" % self.epsilon)
        X, y_org = check_X_y(X, y, accept_sparse=False, dtype=np.float64, 
                order="C")

        y_type = type_of_target(y_org)
        if y_type not in ["binary", "multiclass"]:
            raise ValueError("Label type not allowed for GenSVM: %r" % y_type)

        # This is necessary because GenSVM expects classes to go from 1 to 
        # n_class
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y_org)
        y += 1

        self.coef_, self.intercept_, self.n_iter_, self.n_support_ = \
                _fit_gensvm(X, y, self.p, self.lmd, self.kappa, self.epsilon, 
                        self.weight_idx, self.kernel, self.gamma, self.coef, 
                        self.degree, self.kernel_eigen_cutoff, self.verbose, 
                        self.max_iter, self.random_state)
        return self


    def predict(self, X):
        check_is_fitted(self, "coef_")

        V = np.vstack((self.intercept_, self.coef_))
        predictions = wrapper.predict_wrap(X, V)

        # Transform the classes back to the original form
        predictions -= 1
        outcome = self.encoder.inverse_transform(predictions)

        return outcome
