# -*- coding: utf-8 -*-

"""

    Unit tests for the grid_search module

"""

from __future__ import division, print_function

import numpy as np
import unittest

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    ShuffleSplit,
)
from sklearn.preprocessing import maxabs_scale

from gensvm.gridsearch import (
    GenSVMGridSearchCV,
    _validate_param_grid,
    load_grid_tiny,
    load_grid_small,
    load_grid_full,
)


class GenSVMGridSearchCVTestCase(unittest.TestCase):
    def test_validate_param_grid(self):
        """ GENSVM_GRID: Test parameter grid validation """
        pg = {
            "p": [1, 1.5, 2.0],
            "kappa": [-0.9, 1.0],
            "lmd": [0.1, 1.0],
            "epsilon": [0.01, 0.002],
            "gamma": [1.0, 2.0],
            "weights": ["unit", "group"],
        }

        _validate_param_grid(pg)

        tmp = {k: v for k, v in pg.items()}
        tmp["p"] = [0.5, 1.0, 2.0]
        with self.assertRaises(ValueError):
            _validate_param_grid(tmp)

        tmp = {k: v for k, v in pg.items()}
        tmp["kappa"] = [-1.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            _validate_param_grid(tmp)

        tmp = {k: v for k, v in pg.items()}
        tmp["lmd"] = [-1.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            _validate_param_grid(tmp)

        tmp = {k: v for k, v in pg.items()}
        tmp["epsilon"] = [-1.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            _validate_param_grid(tmp)

        tmp = {k: v for k, v in pg.items()}
        tmp["gamma"] = [-1.0, 0.0, 1.0]
        with self.assertRaises(ValueError):
            _validate_param_grid(tmp)

        tmp = {k: v for k, v in pg.items()}
        tmp["weights"] = ["unit", "group", "other"]
        with self.assertRaises(ValueError):
            _validate_param_grid(tmp)

    def test_fit_predict_strings(self):
        """ GENSVM_GRID: Test fit and predict with string targets """
        iris = load_iris()
        X = iris.data
        y = iris.target
        labels = iris.target_names
        yy = labels[y]
        X_train, X_test, y_train, y_test = train_test_split(X, yy)

        pg = {
            "p": [1, 1.5, 2.0],
            "kappa": [-0.9, 1.0],
            "lmd": [0.1, 1.0],
            "epsilon": [0.01, 0.002],
            "gamma": [1.0, 2.0],
            "weights": ["unit", "group"],
        }

        clf = GenSVMGridSearchCV(pg)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        pred_set = set(y_pred)
        label_set = set(labels)
        self.assertTrue(pred_set.issubset(label_set))

    def test_fit_score(self):
        """ GENSVM_GRID: Test fit and score """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pg = {
            "p": [1, 1.5, 2.0],
            "kappa": [-0.9, 1.0, 5.0],
            "lmd": [pow(2, x) for x in range(-12, 9, 2)],
        }

        clf = GenSVMGridSearchCV(pg)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # low for safety
        self.assertGreaterEqual(score, 0.80)

    def test_refit(self):
        """ GENSVM_GRID: Test refit """
        # we use the fact that large regularization parameters usually don't
        # give a good fit.
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pg = {"lmd": [1e-4, 100, 10000]}

        clf = GenSVMGridSearchCV(pg)
        clf.fit(X_train, y_train)

        self.assertTrue(hasattr(clf, "best_params_"))
        self.assertTrue(clf.best_params_ == {"lmd": 1e-4})

    def test_multimetric(self):
        """ GENSVM_GRID: Test multimetric """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pg = {"p": [1., 1.5, 2.]}

        clf = GenSVMGridSearchCV(
            pg, scoring=["accuracy", "adjusted_rand_score"], refit=False
        )
        clf.fit(X_train, y_train)

        self.assertTrue(clf.multimetric_)
        self.assertTrue("mean_test_accuracy" in clf.cv_results_)
        self.assertTrue("mean_test_adjusted_rand_score" in clf.cv_results_)

    def test_refit_multimetric(self):
        """ GENSVM_GRID: Test refit with multimetric """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pg = {"lmd": [1e-4, 100, 10000]}

        clf = GenSVMGridSearchCV(
            pg, scoring=["accuracy", "adjusted_rand_score"], refit="accuracy"
        )
        clf.fit(X_train, y_train)

        self.assertTrue(hasattr(clf, "best_params_"))
        self.assertTrue(hasattr(clf, "best_estimator_"))
        self.assertTrue(hasattr(clf, "best_index_"))
        self.assertTrue(hasattr(clf, "best_score_"))
        self.assertTrue(clf.best_params_ == {"lmd": 1e-4})

    def test_params_rbf_kernel(self):
        """ GENSVM_GRID: Test best params with RBF kernel """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        pg = {"lmd": [1e-4, 100, 10000], "kernel": ["rbf"]}

        clf = GenSVMGridSearchCV(pg)
        clf.fit(X_train, y_train)

        self.assertTrue(hasattr(clf, "best_params_"))

        y_pred = clf.predict(X_test, trainX=X_train)
        del y_pred

    def test_invalid_y(self):
        """ GENSVM_GRID: Check raises for invalid y type """
        pg = {"lmd": [1e-4, 100, 10000], "kernel": ["rbf"]}
        clf = GenSVMGridSearchCV(pg)
        X = np.random.random((20, 4))
        y = np.random.random((20,))
        with self.assertRaises(ValueError) as err:
            clf.fit(X, y)
        exc = err.exception
        self.assertEqual(
            exc.args, ("Label type not allowed for GenSVM: 'continuous'",)
        )

    def slowtest_gridsearch_warnings(self):
        """ GENSVM_GRID: Check grid search with warnings """
        np.random.seed(123)
        X, y = load_digits(4, return_X_y=True)
        small = {}
        for k in [1, 2, 3]:
            tmp = X[y == k, :]
            small[k] = tmp[np.random.choice(tmp.shape[0], 20), :]

        Xs = np.vstack((small[1], small[2], small[3]))
        ys = np.hstack((np.ones(20), 2 * np.ones(20), 3 * np.ones(20)))
        pg = {
            "p": [1.0, 2.0],
            "lmd": [pow(10, x) for x in range(-4, 1, 2)],
            "epsilon": [1e-6],
        }
        gg = GenSVMGridSearchCV(pg, verbose=True)
        gg.fit(Xs, ys)

    def test_gridsearch_tiny(self):
        """ GENSVM_GRID: Test with tiny grid """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=123
        )

        tiny = load_grid_tiny()
        for x in tiny:
            x["epsilon"] = [1e-5]
        clf = GenSVMGridSearchCV(param_grid=tiny)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        # low threshold on purpose for testing on Travis
        # Real performance should be higher!
        self.assertGreaterEqual(score, 0.70)

    def test_gridsearch_small(self):
        """ GENSVM_GRID: Test with small grid """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=123
        )

        small = load_grid_small()
        small["epsilon"] = [1e-5]
        clf = GenSVMGridSearchCV(param_grid=small)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        # low threshold on purpose for testing on Travis
        # Real performance should be higher!
        self.assertGreaterEqual(score, 0.70)

    def test_gridsearch_full(self):
        """ GENSVM_GRID: Test with full grid """
        X, y = load_iris(return_X_y=True)
        X = maxabs_scale(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=123
        )

        full = load_grid_full()
        full["epsilon"] = [1e-5]
        clf = GenSVMGridSearchCV(param_grid=full)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        # low threshold on purpose for testing on Travis
        # Real performance should be higher!
        self.assertGreaterEqual(score, 0.70)

    def test_gridsearch_stratified(self):
        """ GENSVM_GRID: Error on using shufflesplit """
        X, y = load_iris(return_X_y=True)

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        with self.assertRaises(ValueError):
            GenSVMGridSearchCV(param_grid="tiny", verbose=1, cv=cv)

        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        with self.assertRaises(ValueError):
            GenSVMGridSearchCV(param_grid="tiny", verbose=1, cv=cv)
