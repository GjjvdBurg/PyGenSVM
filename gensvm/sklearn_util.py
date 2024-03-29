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

import numbers
import numpy as np
import warnings

from collections import defaultdict
from contextlib import suppress
from functools import partial

from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.metrics._scorer import check_scoring

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

from numpy.ma import MaskedArray

from sklearn.exceptions import NotFittedError
from sklearn.model_selection._validation import (
    _normalize_score_results,
    _aggregate_score_dicts,
)


def _skl_format_cv_results(
    out,
    return_train_score,
    candidate_params,
    n_candidates,
    n_splits,
    scorers,
    iid,
):

    out = _aggregate_score_dicts(out)

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

        if key_name.startswith(("train_", "test_")) and np.any(
            ~np.isfinite(array_means)
        ):
            warnings.warn(
                f"One or more of the {key_name.split('_')[0]} scores "
                f"are non-finite: {array_means}",
                category=UserWarning,
            )

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

    _store("fit_time", out["fit_time"])
    _store("score_time", out["score_time"])
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

    test_scores_dict = _normalize_score_results(out["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(out["train_scores"])

    for scorer_name in test_scores_dict:
        # Computed the (weighted) mean and std for test scores alone
        _store(
            "test_%s" % scorer_name,
            test_scores_dict[scorer_name],
            splits=True,
            rank=True,
            weights=None,
        )
        if return_train_score:
            _store(
                "train_%s" % scorer_name,
                train_scores_dict[scorer_name],
                splits=True,
            )
    return results


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
        if not hasattr(estimator, "best_estimator_"):
            raise NotFittedError(
                "This %s instance is not fitted yet. Call "
                "'fit' with appropriate arguments before using this "
                "estimator." % type(estimator).__name__
            )


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


def _skl_score(estimator, X_test, y_test, scorer):
    """Compute the score(s) of an estimator on a given test set.
    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(**scorer)
    if y_test is None:
        scores = scorer(estimator, X_test)
    else:
        scores = scorer(estimator, X_test, y_test)

    error_msg = (
        "scoring must return a number, got %s (%s) " "instead. (scorer=%s)"
    )
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _skl_check_refit_for_multimetric(self, scores):
    """Check `refit` is compatible with `scores` is valid"""
    multimetric_refit_msg = (
        "For multi-metric scoring, the parameter refit must be set to a "
        "scorer key or a callable to refit an estimator with the best "
        "parameter setting on the whole data and make the best_* "
        "attributes available for that metric. If this is not needed, "
        f"refit should be set to False explicitly. {self.refit!r} was "
        "passed."
    )

    valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

    if (
        self.refit is not False
        and not valid_refit_dict
        and not callable(self.refit)
    ):
        raise ValueError(multimetric_refit_msg)
