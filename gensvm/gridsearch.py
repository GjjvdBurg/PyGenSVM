# -*- coding: utf-8 -*-

"""Functions for doing an efficient GenSVM grid search

This module contains functions to run a grid search for the GenSVM model. This 
is implemented in a separate class because it uses the GenSVM C library to do 
the actual grid search. The C routines for the grid search use warm starts for 
the computations and are therefore more efficient.

"""

from __future__ import print_function, division

import numpy as np
import time

from operator import itemgetter

from sklearn.base import ClassifierMixin, BaseEstimator, MetaEstimatorMixin
from sklearn.model_selection import (
    ParameterGrid,
    check_cv,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection._validation import _score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import indexable

from .cython_wrapper import wrapper
from .core import GenSVM
from .sklearn_util import (
    _skl_format_cv_results,
    _skl_check_scorers,
    _skl_check_is_fitted,
    _skl_grid_score,
)


def _sort_candidate_params(candidate_params):
    if any(("epsilon" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("epsilon"), reverse=True)
    if any(("p" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("p"))
    if any(("lmd" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("lmd"))
    if any(("kappa" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("kappa"))
    if any(("weights" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("weights"))
    if any(("gamma" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("gamma"))
    if any(("degree" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("degree"))
    if any(("coef" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("coef"))
    if any(("kernel" in p for p in candidate_params)):
        candidate_params.sort(key=itemgetter("kernel"))


def _validate_param_grid(param_grid):
    """Check if the parameter values are valid

    This basically does the same checks as in the constructor of the 
    :class:`core.GenSVM` class, but for the entire parameter grid.

    """
    # the conditions that the parameters must satisfy
    conditions = {
        "p": lambda x: 1.0 <= x <= 2.0,
        "kappa": lambda x: x > -1.0,
        "lmd": lambda x: x > 0,
        "epsilon": lambda x: x > 0,
        "gamma": lambda x: x != 0,
        "weights": lambda x: x in ["unit", "group"],
    }

    for param in conditions:
        if param in param_grid:
            if not all(map(conditions[param], param_grid[param])):
                raise ValueError(
                    "Invalid value in grid for parameter: %s." % (param)
                )


class _MockEstimator(ClassifierMixin):
    # This mock estimator facilitates the use of the Scorer class of
    # Scikit-Learn. Basically, we want to use the _score function of
    # sklearn.model_selection._validation, but we don't keep track of the
    # individual estimators in the GenSVM C grid search code. With this wrapper
    # we can mock an estimator for the _score function.

    # The ClassifierMixin adds the score method to the estimator. This allows us
    # to leave scoring=None as the default to the GenSVMGridSearchCV class and
    # ends up using the accuracy_score metric.

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions


def _wrap_score(y_pred, y_true, scorers, is_multimetric):
    start_time = time.time()
    results = {}
    # we use -1 to signify missing predictions because numpy has no integer NaN
    if np.any(y_pred < 0):
        if is_multimetric:
            for name in scorers:
                results[name] = np.nan
        else:
            results["score"] = np.nan
    else:
        estimator = _MockEstimator(y_pred)
        results = _score(estimator, None, y_true, scorers, is_multimetric)
    score_time = time.time() - start_time
    return results, score_time


def _format_results(
    results,
    cv_idx,
    true_y,
    scorers,
    iid,
    return_train_score=True,
    return_n_test_samples=True,
    return_times=True,
    return_parameters=False,
):
    """Format the results from the grid search

    Parameters
    ----------

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    """

    out = []
    candidate_params = results["params"]
    n_candidates = len(candidate_params)
    n_splits = len(np.unique(cv_idx))

    is_multimetric = not callable(scorers)

    # Out must be a list of dicts of size n_params x n_splits that iterates
    # over the params in the list and for each param iterates over the splits.
    for param, durations, predictions in zip(
        results["params"], results["durations"], results["predictions"]
    ):
        fit_times = durations
        is_missing = np.any(np.isnan(durations))

        for test_idx in sorted(np.unique(cv_idx)):
            ret = []
            score_time = 0

            if return_train_score:
                train_pred = predictions[cv_idx != test_idx,]
                y_train = true_y[cv_idx != test_idx,]
                train_score, score_t = _wrap_score(
                    train_pred, y_train, scorers, is_multimetric
                )
                score_time += score_t
                ret.append(train_score)

            test_pred = predictions[cv_idx == test_idx,]
            y_test = true_y[cv_idx == test_idx,]
            test_score, score_t = _wrap_score(
                test_pred, y_test, scorers, is_multimetric
            )
            score_time += score_t
            ret.append(test_score)

            if return_n_test_samples:
                ret.append(len(y_test))
            if return_times:
                fit_time = fit_times[test_idx]
                score_time = np.nan if is_missing else score_time
                ret.extend([fit_time, score_time])
            if return_parameters:
                ret.append(param)

            out.append(ret)

    cv_results_ = _skl_format_cv_results(
        out,
        return_train_score,
        candidate_params,
        n_candidates,
        n_splits,
        scorers,
        iid,
    )

    return cv_results_


def _fit_grid_gensvm(
    X,
    y,
    groups,
    candidate_params,
    scorers,
    cv,
    refit,
    verbose,
    return_train_score,
    iid,
):
    """Utility function for fitting the grid search for GenSVM

    This function sorts the parameter grid for optimal computation speed, sets 
    the desired verbosity, generates the cross validation splits, and calls the 
    low-level training routine in the Cython wrapper.

    For parameters, see :class:`.GenSVMGridSearchCV`.

    Returns
    -------
    cv_results_ : dict
        The cross validation results. See :func:`~GenSVMGridSearchCV.fit`.

    """

    # sort the candidate params
    # the optimal order of the parameters from inner to outer loop is: epsilon,
    # p, lambda, kappa, weights, kernel, ???
    _sort_candidate_params(candidate_params)

    # set the verbosity in GenSVM
    wrapper.set_verbosity_wrap(verbose)

    # NOTE: The C library can compute the accuracy score and destroy the exact
    # predictions, but this doesn't allow us to compute the score per fold. So
    # we always want to get the raw predictions for each grid point.
    store_predictions = True

    # Convert the cv variable to a cv_idx array
    cv = check_cv(cv, y, classifier=True)
    n_folds = cv.get_n_splits(X, y, groups)
    cv_idx = np.zeros((X.shape[0],), dtype=np.int_) - 1
    fold_idx = 0
    for train, test in cv.split(X, y, groups):
        cv_idx[test,] = fold_idx
        fold_idx += 1

    results_ = wrapper.grid_wrap(
        X,
        y,
        candidate_params,
        int(store_predictions),
        cv_idx,
        int(n_folds),
        int(verbose),
    )
    cv_results_ = _format_results(results_, cv_idx, y, scorers, iid)

    return cv_results_, n_folds


class GenSVMGridSearchCV(BaseEstimator, MetaEstimatorMixin):
    """GenSVM cross validated grid search

    This class implements efficient GenSVM grid search with cross validation.  
    One of the strong features of GenSVM is that seeding the classifier 
    properly can greatly reduce total training time. This class ensures that 
    the grid search is done in the most efficient way possible.

    The implementation of this class is based on the `GridSearchCV`_ class in 
    scikit-learn. The documentation of the various parameters is therefore 
    mostly the same. This is done to provide the user with a familiar and 
    easy-to-use interface to doing a grid search with GenSVM. A separate class 
    was needed to benefit from the fast low-level C implementation of grid 
    search in the GenSVM library.

    Parameters
    ----------
    param_grid : string, dict, or list of dicts
        If a string, it must be either 'tiny', 'small', or 'full' to load the 
        predefined parameter grids (see the functions :func:`load_grid_tiny`, 
        :func:`load_grid_small`, and :func:`load_grid_full`).

        Otherwise, a dictionary of parameter names (strings) as keys and lists 
        of parameter settings to evaluate as values, or a list of such dicts.  
        The GenSVM model will be evaluated at all combinations of the 
        parameters.

    scoring : string, callable, list/tuple, dict or None
        A single string (see :ref:`scoring_parameter`) or a callable (see 
        :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings 
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single 
        value. Metric functions returning a list/array of values can be wrapped 
        into multiple scorers that return one value each. 

        If None, the `accuracy_score`_ is used. 

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across the 
        folds, and the loss minimized is the total loss per sample and not the 
        mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for 
        cv are:

          - None, to use the default 5-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.

        For integer/None inputs, :class:`StratifiedKFold 
        <sklearn.model_selection.StratifiedKFold>` is used.  In all other 
        cases, :class:`KFold <sklearn.model_selection.KFold>` is used.

        Refer to the `scikit-learn User Guide on cross validation`_ for the 
        various strategies that can be used here.

        NOTE: At the moment, the ShuffleSplit and StratifiedShuffleSplit are 
        not supported in this class. If you need these, you can use the GenSVM 
        classifier directly with the GridSearchCV object from scikit-learn.  
        (these methods require significant changes in the low-level library 
        before they can be supported).

    refit : boolean, or string, default=True
        Refit the GenSVM estimator with the best found parameters on the whole 
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the 
        scorer to be used to find the best parameters for refitting the 
        estimator at the end.

        The refitted estimator is made available at the `:attr:best_estimator_ 
        <.GenSVMGridSearchCV.best_estimator_>` attribute and allows the user to 
        use the :func:`~GenSVMGridSearchCV.predict` method directly on this 
        :class:`.GenSVMGridSearchCV` instance.

        Also for multiple metric evaluation, the attributes :attr:`best_index_ 
        <.GenSVMGridSearchCV.best_index_>`, :attr:`best_score_ 
        <.GenSVMGridSearchCV.best_score_>` and :attr:`best_params_ 
        <.GenSVMGridSearchCV:best_params_>` will only be available if ``refit`` 
        is set and all of them will be determined w.r.t this specific scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    return_train_score : boolean, default=True
        If ``False``, the :attr:`cv_results_ <.GenSVMGridSearchCV.cv_results_>` 
        attribute will not include training scores.

    Examples
    --------
    >>> from gensvm import GenSVMGridSearchCV
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> param_grid = {'p': [1.0, 2.0], 'kappa': [-0.9, 0.0, 1.0]}
    >>> clf = GenSVMGridSearchCV(param_grid)
    >>> clf.fit(iris.data, iris.target)
    GenSVMGridSearchCV(cv=None, iid=True,
          param_grid={'p': [1.0, 2.0], 'kappa': [-0.9, 0.0, 1.0]},
          refit=True, return_train_score=True, scoring=None, verbose=0)

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be 
        imported into a pandas `DataFrame`_.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |        0.8      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |        0.7      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
            'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
            'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE:

        The key ``'params'`` is used to store a list of parameter settings 
        dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and 
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are 
        available in the :attr:`cv_results_ <.GenSVMGridSearchCV.cv_results_>` 
        dict at the keys ending with that scorer's name (``'_<scorer_name>'``) 
        instead of ``'_score'`` shown above. ('split0_test_precision', 
        'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator which gave 
        highest score (or smallest loss if specified) on the left out data. Not 
        available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator 

        For multi-metric evaluation, this is present only if ``refit`` is 
        specified.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data. 

        For multi-metric evaluation, this is present only if ``refit`` is 
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best 
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives 
        the parameter setting for the best model, that gives the highest mean 
        score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is 
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best parameters 
        for the model.

        For multi-metric evaluation, this attribute holds the validated 
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the left out 
    data, unless an explicit score is passed in which case it is used instead.

    See Also
    --------
    `ParameterGrid`_:
        Generates all the combinations of a hyperparameter grid.

    :class:`.GenSVM`:
        The GenSVM classifier

    .. _GridSearchCV:
        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    .. _accuracy_score:
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    .. _scikit-learn User Guide on cross validation:
        http://scikit-learn.org/stable/modules/cross_validation.html

    .. _ParameterGrid:
        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
    .. _DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
    """

    def __init__(
        self,
        param_grid="tiny",
        scoring=None,
        iid=True,
        cv=None,
        refit=True,
        verbose=0,
        return_train_score=True,
    ):

        self.param_grid = param_grid
        if isinstance(self.param_grid, str):
            if self.param_grid == "tiny":
                self.param_grid = load_grid_tiny()
            elif self.param_grid == "small":
                self.param_grid = load_grid_small()
            elif self.param_grid == "full":
                self.param_grid = load_grid_full()
            else:
                raise ValueError("Unknown param grid %r" % self.param_grid)
        _check_param_grid(self.param_grid)
        _validate_param_grid(self.param_grid)

        self.scoring = scoring
        self.cv = 5 if cv is None else cv
        if isinstance(self.cv, ShuffleSplit) or isinstance(
            self.cv, StratifiedShuffleSplit
        ):
            raise ValueError(
                "ShuffleSplit and StratifiedShuffleSplit are not supported at the moment. Please see the documentation for more info"
            )
        self.refit = refit
        self.verbose = verbose
        self.return_train_score = return_train_score
        self.iid = iid

    def _get_param_iterator(self):
        return ParameterGrid(self.param_grid)

    def fit(self, X, y, groups=None):
        """Run GenSVM grid search with all sets of parameters

        Parameters
        ----------

        X : array-like, shape = (n_samples, n_features)
            Training data, where n_samples is the number of observations and 
            n_features is the number of features.

        y : array-like, shape = (n_samples, )
            Target vector for the training data.

        groups : array-like, with shape (n_samples, ), optional
            Group labels for the samples used while splitting the dataset into 
            train/test sets.

        Returns
        -------
        self : object
            Return self.

        """

        X, y_orig = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C"
        )

        y_type = type_of_target(y_orig)
        if y_type not in ["binary", "multiclass"]:
            raise ValueError("Label type not allowed for GenSVM: %r" % y_type)

        # This is necessary because GenSVM expects classes to go from 1 to
        # n_class
        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y_orig)
        y += 1

        candidate_params = list(self._get_param_iterator())

        scorers, self.multimetric_, refit_metric = _skl_check_scorers(
            self.scoring, self.refit
        )

        X, y, groups = indexable(X, y, groups)

        results, n_splits = _fit_grid_gensvm(
            X,
            y,
            groups,
            candidate_params,
            scorers,
            self.cv,
            self.refit,
            self.verbose,
            self.return_train_score,
            self.iid,
        )

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = candidate_params[self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_
            ]

        if self.refit:
            # when we used a nonlinear kernel and specified no gamma, then
            # gamma='auto' was used. We need to save the actual numerical value
            # for use in the predict method later on, so we extract that here.
            if (
                "kernel" in self.best_params_
                and not self.best_params_["kernel"] == "linear"
                and not "gamma" in self.best_params_
            ):
                self.best_params_["gamma"] = 1. / X.shape[1]
            self.best_estimator_ = GenSVM(**self.best_params_)
            # y_orig because GenSVM fit must know the conversion for predict to
            # work correctly
            self.best_estimator_.fit(X, y_orig)

        ## Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers["score"]

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def score(self, X, y):
        """Compute the score on the test data given the true labels

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data, where n_samples is the number of observations and 
            n_features is the number of features.

        y : array-like, shape = (n_samples, )
            True labels for the test data.


        Returns
        -------
        score : float

        """
        _skl_check_is_fitted(self, "score", self.refit)
        return _skl_grid_score(
            X,
            y,
            self.scorer_,
            self.best_estimator_,
            self.refit,
            self.multimetric_,
        )

    def predict(self, X, trainX=None):
        """Predict the class labels on the test data

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data, where n_samples is the number of observations and 
            n_features is the number of features.

        trainX : array, shape = [n_train_samples, n_features]
            Only for nonlinear prediction with kernels: the training data used 
            to train the model.

        Returns
        -------
        y_pred : array-like, shape = (n_samples, )
            Predicted class labels of the data in X.

        """
        _skl_check_is_fitted(self, "predict", self.refit)
        return self.best_estimator_.predict(X, trainX=trainX)


def load_grid_tiny():
    """ Load a tiny parameter grid for the GenSVM grid search

    This function returns a parameter grid to use in the GenSVM grid search.  
    This grid was obtained by analyzing the experiments done for the GenSVM 
    paper and selecting the configurations that achieve accuracy within the 
    95th percentile on over 90% of the datasets. It is a good start for a 
    parameter search with a reasonably high chance of achieving good 
    performance on most datasets.

    Note that this grid is only tested to work well in combination with the 
    linear kernel.

    Returns
    -------

    pg : list
        List of 10 parameter configurations that are likely to perform 
        reasonably well.

    """

    pg = [
        {"p": [2.0], "kappa": [5.0], "lmd": [pow(2, -16)], "weights": ["unit"]},
        {"p": [2.0], "kappa": [5.0], "lmd": [pow(2, -18)], "weights": ["unit"]},
        {"p": [2.0], "kappa": [0.5], "lmd": [pow(2, -18)], "weights": ["unit"]},
        {
            "p": [2.0],
            "kappa": [5.0],
            "lmd": [pow(2, -18)],
            "weights": ["group"],
        },
        {
            "p": [2.0],
            "kappa": [-0.9],
            "lmd": [pow(2, -18)],
            "weights": ["unit"],
        },
        {"p": [2.0], "kappa": [5.0], "lmd": [pow(2, -14)], "weights": ["unit"]},
        {
            "p": [2.0],
            "kappa": [0.5],
            "lmd": [pow(2, -18)],
            "weights": ["group"],
        },
        {
            "p": [1.5],
            "kappa": [-0.9],
            "lmd": [pow(2, -18)],
            "weights": ["unit"],
        },
        {"p": [2.0], "kappa": [0.5], "lmd": [pow(2, -16)], "weights": ["unit"]},
        {
            "p": [2.0],
            "kappa": [0.5],
            "lmd": [pow(2, -16)],
            "weights": ["group"],
        },
    ]
    return pg


def load_grid_small():
    """Load a small parameter grid for GenSVM

    This function loads a default parameter grid to use for the #' GenSVM 
    gridsearch. It contains all possible combinations of the following #' 
    parameter sets::

        pg = {
            'p': [1.0, 1.5, 2.0],
            'lmd': [1e-8, 1e-6, 1e-4, 1e-2, 1],
            'kappa': [-0.9, 0.5, 5.0],
            'weights': ['unit', 'group'],
        }

    Returns
    -------

    pg : dict
        Mapping from parameters to lists of values for those parameters. To be 
        used as input for the :class:`.GenSVMGridSearchCV` class.
    """
    pg = {
        "p": [1.0, 1.5, 2.0],
        "lmd": [1e-8, 1e-6, 1e-4, 1e-2, 1],
        "kappa": [-0.9, 0.5, 5.0],
        "weights": ["unit", "group"],
    }
    return pg


def load_grid_full():
    """Load the full parameter grid for GenSVM

    This is the parameter grid used in the GenSVM paper to run the grid search 
    experiments. It uses a large grid for the ``lmd`` regularization parameter 
    and converges with a stopping criterion of ``1e-8``. This is a relatively 
    small stopping criterion and in practice good classification results can be 
    obtained by using a larger stopping criterion.

    The function returns the following grid::

        pg = {
                'lmd': [pow(2, x) for x in range(-18, 19, 2)],
                'kappa': [-0.9, 0.5, 5.0],
                'p': [1.0, 1.5, 2.0],
                'weights': ['unit', 'group'],
                'epsilon': [1e-8],
                'kernel': ['linear']
             }

    Returns
    -------
    pg : dict
        Mapping from parameters to lists of values for those parameters. To be 
        used as input for the :class:`.GenSVMGridSearchCV` class.
    """
    pg = {
        "lmd": [pow(2, x) for x in range(-18, 19, 2)],
        "kappa": [-0.9, 0.5, 5.0],
        "p": [1.0, 1.5, 2.0],
        "weights": ["unit", "group"],
        "epsilon": [1e-8],
        "kernel": ["linear"],
    }
    return pg
