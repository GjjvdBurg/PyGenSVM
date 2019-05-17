# -*- coding: utf-8 -*-

"""Core functionality for fitting the GenSVM classifier

This module contains the basic definitions to fit a single GenSVM model.

"""

from __future__ import print_function, division

import numpy as np
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from .cython_wrapper import wrapper


def _fit_gensvm(
    X,
    y,
    n_class,
    p,
    lmd,
    kappa,
    epsilon,
    weights,
    sample_weight,
    kernel,
    gamma,
    coef,
    degree,
    kernel_eigen_cutoff,
    verbose,
    max_iter,
    random_state=None,
    seed_V=None,
):

    # process the random state
    rnd = check_random_state(random_state)

    # set the verbosity in GenSVM
    wrapper.set_verbosity_wrap(verbose)

    # convert the weight index
    weight_idx = {"raw": 0, "unit": 1, "group": 2}[weights]

    # run the actual training
    raw_coef_, n_SV_, n_iter_, training_error_, status_ = wrapper.train_wrap(
        X,
        y,
        n_class,
        p,
        lmd,
        kappa,
        epsilon,
        weight_idx,
        sample_weight,
        kernel,
        gamma,
        coef,
        degree,
        kernel_eigen_cutoff,
        max_iter,
        rnd.randint(np.iinfo("i").max),
        seed_V,
    )

    # process output
    if status_ == 1 and verbose > 0:
        warnings.warn(
            "GenSVM optimization prematurely ended due to a "
            "incorrect step in the optimization algorithm. "
            "This can be due to data quality issues, hyperparameter "
            "settings, or numerical precision errors.",
            FitFailedWarning,
        )

    if status_ == 2 and verbose > 0:
        warnings.warn(
            "GenSVM failed to converge, you may want to increase "
            "the number of iterations.",
            ConvergenceWarning,
        )

    coef_ = raw_coef_[1:, :]
    intercept_ = raw_coef_[0, :]

    return coef_, intercept_, n_iter_, n_SV_


class GenSVM(BaseEstimator, ClassifierMixin):
    """Generalized Multiclass Support Vector Machine Classification.

    This class implements the basic GenSVM classifier. GenSVM is a generalized 
    multiclass SVM which is flexible in the weighting of misclassification 
    errors. It is this flexibility that makes it perform well on diverse 
    datasets.

    The :func:`~GenSVM.fit` and :func:`~GenSVM.predict` methods of this class 
    use the GenSVM C library for the actual computations.

    Parameters
    ----------
    p : float, optional (default=1.0)
        Parameter for the L_p norm of the loss function (1.0 <= p <= 2.0)

    lmd : float, optional (default=1e-5)
        Parameter for the regularization term of the loss function (lmd > 0)

    kappa : float, optional (default=0.0)
        Parameter for the hinge function in the loss function (kappa > -1.0)

    weights: string, optional (default='unit')
        Type of sample weights to use. Options are 'unit' for unit weights and 
        'group' for group size correction weights (equation 4 in the paper).

        It is also possible to provide an explicit vector of sample weights 
        through the :func:`~GenSVM.fit` method. If so, it will override the 
        setting provided here.

    kernel : string, optional (default='linear')
        Specify the kernel type to use in the classifier. It must be one of 
        'linear', 'poly', 'rbf', or 'sigmoid'.

    gamma : float, optional (default='auto')
        Kernel parameter for the rbf, poly, and sigmoid kernel. If gamma is 
        'auto' then 1/n_features will be used. See `Kernels in GenSVM 
        <gensvm_kernels_>`_ for the exact implementation of the kernels.

    coef : float, optional (default=1.0)
        Kernel parameter for the poly and sigmoid kernel. See `Kernels in 
        GenSVM <gensvm_kernels_>`_ for the exact implementation of the kernels.

    degree : float, optional (default=2.0)
        Kernel parameter for the poly kernel. See `Kernels in GenSVM 
        <gensvm_kernels_>`_ for the exact implementation of the kernels.

    kernel_eigen_cutoff : float, optional (default=1e-8)
        Cutoff point for the reduced eigendecomposition used with nonlinear 
        GenSVM.  Eigenvectors for which the ratio between their corresponding 
        eigenvalue and the largest eigenvalue is smaller than the cutoff will 
        be dropped.

    verbose : int, (default=0)
        Enable verbose output

    random_state : None, int, instance of RandomState
        The seed for the random number generation used for initialization where 
        necessary. See the documentation of 
        ``sklearn.utils.check_random_state`` for more info.

    max_iter : int, (default=1e8)
        The maximum number of iterations to be run.


    Attributes
    ----------
    coef_ : array, shape = [n_features, n_classes-1]
        Weights assigned to the features (coefficients in the primal problem)

    intercept_ : array, shape = [n_classes-1]
        Constants in the decision function

    combined_coef_ : array, shape = [n_features+1, n_classes-1]
        Combined weights matrix for the seed_V parameter to the fit method

    n_iter_ : int
        The number of iterations that were run during training.

    n_support_ : int
        The number of support vectors that were found


    See Also
    --------
    :class:`.GenSVMGridSearchCV`:
        Helper class to run an efficient grid search for GenSVM.


    .. _gensvm_kernels:
        https://gensvm.readthedocs.io/en/latest/#kernels-in-gensvm

    """

    def __init__(
        self,
        p=1.0,
        lmd=1e-5,
        kappa=0.0,
        epsilon=1e-6,
        weights="unit",
        kernel="linear",
        gamma="auto",
        coef=1.0,
        degree=2.0,
        kernel_eigen_cutoff=1e-8,
        verbose=0,
        random_state=None,
        max_iter=1e8,
    ):

        if not 1.0 <= p <= 2.0:
            raise ValueError(
                "Value for p should be within [1, 2]; got p = %r" % p
            )
        if not kappa > -1.0:
            raise ValueError(
                "Value for kappa should be larger than -1; got "
                "kappa = %r" % kappa
            )
        if not lmd > 0:
            raise ValueError(
                "Value for lmd should be larger than 0; got " "lmd = %r" % lmd
            )
        if not epsilon > 0:
            raise ValueError(
                "Value for epsilon should be larger than 0; got "
                "epsilon = %r" % epsilon
            )
        if gamma == 0.0:
            raise ValueError("A gamma value of 0.0 is invalid")
        if not weights in ("unit", "group"):
            raise ValueError(
                "Unknown weight parameter specified. Should be "
                "'unit' or 'group'; got %r" % weights
            )
        if not kernel in ("linear", "rbf", "poly", "sigmoid"):
            raise ValueError(
                "Unknown kernel specified. Should be "
                "'linear', 'rbf', 'poly', or 'sigmoid'; got %r" % kernel
            )

        self.p = p
        self.lmd = lmd
        self.kappa = kappa
        self.epsilon = epsilon
        self.weights = weights
        self.kernel = kernel
        self.gamma = gamma
        self.coef = coef
        self.degree = degree
        self.kernel_eigen_cutoff = kernel_eigen_cutoff
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None, seed_V=None):
        """Fit the GenSVM model on the given data

        The model can be fit with or without a seed matrix (``seed_V``). This 
        can be used to provide warm starts for the algorithm.

        Parameters
        ----------

        X : array, shape = (n_observations, n_features)
            The input data. It is expected that only numeric data is given.

        y : array, shape = (n_observations, )
            The label vector, labels can be numbers or strings.

        sample_weight : array, shape = (n_observations, )
            Array of weights that are assigned to individual samples. If not 
            provided, then the weight specification in the constructor is used 
            ('unit' or 'group').

        seed_V : array, shape = (n_features+1, n_classes-1), optional
            Seed coefficient array to use as a warm start for the optimization.  
            It can for instance be the :attr:`combined_coef_ 
            <.GenSVM.combined_coef_>` attribute of a different GenSVM model.  
            This is only supported for the linear kernel.

            NOTE: the size of the seed_V matrix is ``n_features+1`` by 
            ``n_classes - 1``.  The number of columns of ``seed_V`` is leading 
            for the number of classes in the model. For example, if ``y`` 
            contains 3 different classes and ``seed_V`` has 3 columns, we 
            assume that there are actually 4 classes in the problem but one 
            class is just represented in this training data. This can be useful 
            for problems were a certain class has only a few samples.


        Returns
        -------
        self : object
            Returns self.

        """
        X, y_org = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C"
        )
        if not sample_weight is None:
            sample_weight = check_array(
                sample_weight,
                accept_sparse=False,
                ensure_2d=False,
                dtype=np.float64,
                order="C",
            )
            if not len(sample_weight) == X.shape[0]:
                raise ValueError(
                    "sample weight array must have the same number of observations as X"
                )
            weights = "raw"
        else:
            weights = self.weights

        y_type = type_of_target(y_org)
        if y_type not in ["binary", "multiclass"]:
            raise ValueError("Label type not allowed for GenSVM: %r" % y_type)

        if self.gamma == "auto":
            gamma = 1 / X.shape[1]
        else:
            gamma = self.gamma

        # This is necessary because GenSVM expects classes to go from 1 to
        # n_class
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y_org)
        y += 1

        n_class = len(np.unique(y))
        if not seed_V is None and self.kernel != "linear":
            warnings.warn(
                "Warm starts are only supported for the "
                "linear kernel. The seed_V parameter will be ignored."
            )
            seed_V = None
        if not seed_V is None:
            n_samples, n_features = X.shape
            if seed_V.shape[1] + 1 > n_class:
                n_class = seed_V.shape[1]
            if seed_V.shape[0] - 1 != n_features or (
                seed_V.shape[1] + 1 < n_class
            ):
                raise ValueError(
                    "Seed V must have shape [%i, %i], "
                    "but has shape [%i, %i]"
                    % (
                        n_features + 1,
                        n_class - 1,
                        seed_V.shape[0],
                        seed_V.shape[1],
                    )
                )

        self.coef_, self.intercept_, self.n_iter_, self.n_support_ = _fit_gensvm(
            X,
            y,
            n_class,
            self.p,
            self.lmd,
            self.kappa,
            self.epsilon,
            weights,
            sample_weight,
            self.kernel,
            gamma,
            self.coef,
            self.degree,
            self.kernel_eigen_cutoff,
            self.verbose,
            self.max_iter,
            self.random_state,
            seed_V,
        )
        return self

    def predict(self, X, trainX=None):
        """Predict the class labels on the given data

        Parameters
        ----------
        X : array, shape = [n_test_samples, n_features]
            Data for which to predict the labels

        trainX : array, shape = [n_train_samples, n_features]
            Only for nonlinear prediction with kernels: the training data used 
            to train the model.

        Returns
        -------
        y_pred : array, shape = (n_samples, )
            Predicted class labels of the data in X.

        """

        if (not self.kernel == "linear") and trainX is None:
            raise ValueError(
                "Training data must be provided with nonlinear prediction"
            )
        if not trainX is None and not X.shape[1] == trainX.shape[1]:
            raise ValueError(
                "Test and training data should have the same number of features"
            )

        # make sure arrays are C-contiguous
        X = check_array(X, accept_sparse=False, dtype=np.float64, order="C")
        if not trainX is None:
            trainX = check_array(
                trainX, accept_sparse=False, dtype=np.float64, order="C"
            )

        gamma = 1.0 / X.shape[1] if self.gamma == "auto" else self.gamma

        V = self.combined_coef_
        if self.kernel == "linear":
            predictions = wrapper.predict_wrap(X, V)
        else:
            n_class = len(self.encoder.classes_)
            kernel_idx = wrapper.GENSVM_KERNEL_TYPES.index(self.kernel)
            predictions = wrapper.predict_kernels_wrap(
                X,
                trainX,
                V,
                n_class,
                kernel_idx,
                gamma,
                self.coef,
                self.degree,
                self.kernel_eigen_cutoff,
            )

        # Transform the classes back to the original form
        predictions -= 1
        outcome = self.encoder.inverse_transform(predictions)

        return outcome

    @property
    def combined_coef_(self):
        check_is_fitted(self, "coef_")
        check_is_fitted(self, "intercept_")
        return np.vstack((self.intercept_, self.coef_))
