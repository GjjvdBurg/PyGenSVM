# -*- coding: utf-8 -*-

"""Functions in GenSVM that are taken from Scikit-Learn

The GenSVM Python package is designed to work in the same way as Scikit-Learn 
classifiers, as this makes it easier for people familiar with Scikit-Learn to 
use GenSVM. As such, some of the functionality of the GenSVM Python package is 
similar to code in the Scikit-Learn package (such as formatting the grid search 
results). To keep a clean separation between code from Scikit-Learn (which is 
licensed under the BSD license) and code written by the author(s) of the GenSVM 
package, the code from scikit-learn is placed here in explicit self-contained 
functions. To comply with clause a of the BSD license, it is repeated below as 
required.

"""

import numpy as np

from collections import defaultdict
from functools import partial

from .core import GenSVM
from .util import get_ranks


# BEGIN SCIKIT LEARN CODE

"""

New BSD License

Copyright (c) 2007–2017 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""

from sklearn.exceptions import NotFittedError
from sklearn.externals import six
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import check_is_fitted


def _skl_format_cv_results(
    out,
    return_train_score,
    candidate_params,
    n_candidates,
    n_splits,
    scorers,
    iid,
):

    # if one choose to see train score, "out" will contain train score info
    if return_train_score:
        (
            train_score_dicts,
            test_score_dicts,
            test_sample_counts,
            fit_time,
            score_time,
        ) = zip(*out)
    else:
        (test_score_dicts, test_sample_counts, fit_time, score_time) = zip(*out)

    # test_score_dicts and train_score dicts are lists of dictionaries and
    # we make them into dict of lists
    test_scores = _aggregate_score_dicts(test_score_dicts)
    if return_train_score:
        train_scores = _aggregate_score_dicts(train_score_dicts)

    results = dict()

    def _store(key_name, array, weights=None, splits=False, rank=False):
        """A small helper to store the scores/times to the cv_results_"""
        # When iterated first by splits, then by parameters
        # We want `array` to have `n_candidates` rows and `n_splits` cols.
        array = np.array(array, dtype=np.float64).reshape(
            n_candidates, n_splits
        )
        if splits:
            for split_i in range(n_splits):
                # Uses closure to alter the results
                results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

        array_means = np.average(array, axis=1, weights=weights)
        results["mean_%s" % key_name] = array_means
        # Weighted std is not directly available in numpy
        array_stds = np.sqrt(
            np.average(
                (array - array_means[:, np.newaxis]) ** 2,
                axis=1,
                weights=weights,
            )
        )
        results["std_%s" % key_name] = array_stds

        if rank:
            results["rank_%s" % key_name] = np.asarray(
                get_ranks(-array_means), dtype=np.int32
            )

    _store("fit_time", fit_time)
    _store("score_time", score_time)
    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(
        partial(MaskedArray, np.empty(n_candidates), mask=True, dtype=object)
    )
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            # An all masked empty array gets created for the key
            # `"param_%s" % name` at the first occurence of `name`.
            # Setting the value at an index also unmasks that index
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)
    # Store a list of param dicts at the key 'params'
    results["params"] = candidate_params

    # NOTE test_sample counts (weights) remain the same for all candidates
    test_sample_counts = np.array(test_sample_counts[:n_splits], dtype=np.int)
    for scorer_name in scorers.keys():
        # Computed the (weighted) mean and std for test scores alone
        _store(
            "test_%s" % scorer_name,
            test_scores[scorer_name],
            splits=True,
            rank=True,
            weights=test_sample_counts if iid else None,
        )
        if return_train_score:
            _store(
                "train_%s" % scorer_name, train_scores[scorer_name], splits=True
            )

    return results


def _skl_check_scorers(scoring, refit):

    scorers, multimetric_ = _check_multimetric_scoring(
        GenSVM(), scoring=scoring
    )
    if multimetric_:
        if refit is not False and (
            not isinstance(refit, six.string_types)
            or
            # This will work for both dict / list (tuple)
            refit not in scorers
        ):
            raise ValueError(
                "For multi-metric scoring, the parameter "
                "refit must be set to a scorer key "
                "to refit an estimator with the best "
                "parameter setting on the whole data and "
                "make the best_* attributes "
                "available for that metric. If this is not "
                "needed, refit should be set to False "
                "explicitly. %r was passed." % refit
            )
        else:
            refit_metric = refit
    else:
        refit_metric = "score"

    return scorers, multimetric_, refit_metric


def _skl_check_is_fitted(estimator, method_name, refit):
    if not refit:
        raise NotFittedError(
            "This %s instance was initialized "
            "with refit=False. %s is "
            "available only after refitting on the best "
            "parameters. You can refit an estimator "
            "manually using the ``best_parameters_`` "
            "attribute" % (type(estimator).__name__, method_name)
        )
    else:
        check_is_fitted(estimator, "best_estimator_")


def _skl_grid_score(X, y, scorer_, best_estimator_, refit, multimetric_):
    """Returns the score on the given data, if the estimator has been 
        refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
    if scorer_ is None:
        raise ValueError(
            "No score function explicitly defined, "
            "and the estimator doesn't provide one %s" % best_estimator_
        )
    score = scorer_[refit] if multimetric_ else scorer_
    return score(best_estimator_, X, y)
