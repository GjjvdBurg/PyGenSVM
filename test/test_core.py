# -*- coding: utf-8 -*-

"""

    Unit tests for the GenSVM class

"""

from __future__ import division, print_function

import numpy as np
import unittest
import warnings

from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import maxabs_scale

from gensvm import GenSVM


class GenSVMTestCase(unittest.TestCase):
    def test_init_1(self):
        """ GENSVM: Sanity check for __init__ """
        clf = GenSVM()
        self.assertEqual(clf.p, 1.0)
        self.assertEqual(clf.lmd, 1e-5)
        self.assertEqual(clf.kappa, 0.0)
        self.assertEqual(clf.epsilon, 1e-6)
        self.assertEqual(clf.weights, "unit")
        self.assertEqual(clf.kernel, "linear")
        self.assertEqual(clf.gamma, "auto")
        self.assertEqual(clf.coef, 1.0)
        self.assertEqual(clf.degree, 2.0)
        self.assertEqual(clf.kernel_eigen_cutoff, 1e-8)
        self.assertEqual(clf.verbose, 0)
        self.assertEqual(clf.random_state, None)
        self.assertEqual(clf.max_iter, 1e8)

    def test_init_invalid_p(self):
        """ GENSVM: Check raises for invalid p """
        with self.assertRaises(ValueError) as err1:
            GenSVM(p=2.6)
        exc1 = err1.exception
        self.assertEqual(
            exc1.args, ("Value for p should be within [1, 2]; got p = 2.6",)
        )
        with self.assertRaises(ValueError) as err2:
            GenSVM(p=0.3)
        exc2 = err2.exception
        self.assertEqual(
            exc2.args, ("Value for p should be within [1, 2]; got p = 0.3",)
        )

    def test_init_invalid_kappa(self):
        """ GENSVM: Check raises for invalid kappa """
        with self.assertRaises(ValueError) as err1:
            GenSVM(kappa=-1.0)
        exc1 = err1.exception
        self.assertEqual(
            exc1.args,
            ("Value for kappa should be larger " "than -1; got kappa = -1.0",),
        )

    def test_init_invalid_lmd(self):
        """ GENSVM: Check raises for invalid lmd """
        with self.assertRaises(ValueError) as err1:
            GenSVM(lmd=-3.0)
        exc1 = err1.exception
        self.assertEqual(
            exc1.args,
            ("Value for lmd should be larger than 0; got lmd = -3.0",),
        )

    def test_init_invalid_epsilon(self):
        """ GENSVM: Check raises for invalid epsilon """
        with self.assertRaises(ValueError) as err1:
            GenSVM(epsilon=-1.0)
        exc1 = err1.exception
        self.assertEqual(
            exc1.args,
            (
                "Value for epsilon should be larger than 0; got "
                "epsilon = -1.0",
            ),
        )

    def test_init_invalid_gamma(self):
        """ GENSVM: Check raises for invalid gamma """
        with self.assertRaises(ValueError) as err1:
            GenSVM(gamma=0.0)
        exc1 = err1.exception
        self.assertEqual(exc1.args, ("A gamma value of 0.0 is invalid",))

    def test_init_invalid_weights(self):
        """ GENSVM: Check raises for invalid weights """
        with self.assertRaises(ValueError) as err1:
            GenSVM(weights="other")
        exc1 = err1.exception
        self.assertEqual(
            exc1.args,
            (
                "Unknown weight parameter specified. Should be "
                "'unit' or 'group'; got 'other'",
            ),
        )

    def test_init_invalid_kernel(self):
        """ GENSVM: Check raises for invalid kernel """
        with self.assertRaises(ValueError) as err1:
            GenSVM(kernel="other")
        exc1 = err1.exception
        self.assertEqual(
            exc1.args,
            (
                "Unknown kernel specified. Should be "
                "'linear', 'rbf', 'poly', or 'sigmoid'; got 'other'",
            ),
        )

    def test_invalid_y(self):
        """ GENSVM: Check raises for invalid y type """
        clf = GenSVM()
        X = np.random.random((20, 4))
        y = np.random.random((20,))
        with self.assertRaises(ValueError) as err:
            clf.fit(X, y)
        exc = err.exception
        self.assertEqual(
            exc.args, ("Label type not allowed for GenSVM: 'continuous'",)
        )

    def test_invalid_seed_V_linear(self):
        """ GENSVM: Test for invalid seed_V shape """
        clf = GenSVM()
        X = np.random.random((20, 4))
        y = np.random.randint(1, 4, (20,))
        seed_V = np.random.random((10, 2))
        with self.assertRaises(ValueError) as err:
            clf.fit(X, y, seed_V=seed_V)
        exc = err.exception
        self.assertEqual(
            exc.args, ("Seed V must have shape [5, 2], but has shape [10, 2]",)
        )

    def test_invalid_seed_V_nonlinear(self):
        """ GENSVM: Test for seed_V for nonlinear kernels """
        clf = GenSVM(kernel="rbf")
        X = np.random.random((20, 4))
        y = np.random.randint(1, 4, (20,))
        seed_V = np.random.random((5, 2))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(X, y, seed_V=seed_V)
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
        self.assertEqual(
            msg,
            (
                "Warm starts are only supported for the linear kernel. "
                "The seed_V parameter will be ignored."
            ),
        )

    def test_fit_predict_strings(self):
        """ GENSVM: Test fit and predict with string targets """
        digits = load_digits(4)
        n_samples = len(digits.images)
        X = digits.images.reshape(n_samples, -1)
        y = digits.target
        labels = np.array(["zero", "one", "two", "three"])
        yy = labels[y]

        X_train, X_test, y_train, y_test = train_test_split(X, yy)
        clf = GenSVM(epsilon=1e-3)  # faster testing
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        pred_set = set(y_pred)
        label_set = set(labels)
        self.assertTrue(pred_set.issubset(label_set))

    def test_fit_nonlinear(self):
        """ GENSVM: Fit and predict with nonlinear kernel """
        data = load_iris()
        X = data.data
        y = data.target_names[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=123
        )
        clf = GenSVM(kernel="rbf", gamma=10, max_iter=5000, random_state=123)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test, trainX=X_train)
        self.assertTrue(
            set(pred).issubset(set(["versicolor", "virginica", "setosa"]))
        )

    def test_fit_nonlinear_auto(self):
        """ GENSVM: Fit and predict with nonlinear kernel """
        data = load_iris()
        X = data.data
        y = data.target_names[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=123
        )
        clf = GenSVM(kernel="rbf", max_iter=1000, random_state=123)
        clf.fit(X_train, y_train)

        pred = clf.predict(X_test, trainX=X_train)
        self.assertTrue(
            set(pred).issubset(set(["versicolor", "virginica", "setosa"]))
        )

    def test_fit_with_seed(self):
        """ GENSVM: Test fit with seeding """

        # This is based on the unit test for gensvm_train in the C library

        n_obs = 10
        n_var = 3
        n_class = 4

        X = np.zeros((n_obs, n_var + 1))
        X[0, 0] = 1.0000000000000000
        X[0, 1] = 0.8056271362589000
        X[0, 2] = 0.4874175854113872
        X[0, 3] = 0.4453015882771756
        X[1, 0] = 1.0000000000000000
        X[1, 1] = 0.7940590105180981
        X[1, 2] = 0.1861049005485224
        X[1, 3] = 0.8469394287449229
        X[2, 0] = 1.0000000000000000
        X[2, 1] = 0.0294257611061681
        X[2, 2] = 0.0242717976065267
        X[2, 3] = 0.5039128672814752
        X[3, 0] = 1.0000000000000000
        X[3, 1] = 0.1746563833537603
        X[3, 2] = 0.9135736087631979
        X[3, 3] = 0.5270258081021366
        X[4, 0] = 1.0000000000000000
        X[4, 1] = 0.0022298761599785
        X[4, 2] = 0.3773482059713607
        X[4, 3] = 0.8009654729622842
        X[5, 0] = 1.0000000000000000
        X[5, 1] = 0.6638830667081945
        X[5, 2] = 0.6467607601353914
        X[5, 3] = 0.0434948735457108
        X[6, 0] = 1.0000000000000000
        X[6, 1] = 0.0770493004546461
        X[6, 2] = 0.3699566427075194
        X[6, 3] = 0.7863539761080217
        X[7, 0] = 1.0000000000000000
        X[7, 1] = 0.2685233952731509
        X[7, 2] = 0.8539966432782011
        X[7, 3] = 0.0967159557826836
        X[8, 0] = 1.0000000000000000
        X[8, 1] = 0.1163951898554611
        X[8, 2] = 0.7667861436369238
        X[8, 3] = 0.5031912600213351
        X[9, 0] = 1.0000000000000000
        X[9, 1] = 0.2290251898688216
        X[9, 2] = 0.4401981048538806
        X[9, 3] = 0.0884616753393881

        X = X[:, 1:]
        y = np.array([2, 1, 3, 2, 3, 2, 4, 1, 3, 4])

        seed_V = np.zeros((n_var + 1, n_class - 1))
        seed_V[0, 0] = 0.8233234072519983
        seed_V[0, 1] = 0.7701104553132680
        seed_V[0, 2] = 0.1102697774064020
        seed_V[1, 0] = 0.7956168453294307
        seed_V[1, 1] = 0.3267543833513200
        seed_V[1, 2] = 0.8659836346403005
        seed_V[2, 0] = 0.5777227081256917
        seed_V[2, 1] = 0.3693175185473680
        seed_V[2, 2] = 0.2728942849022845
        seed_V[3, 0] = 0.4426030703804438
        seed_V[3, 1] = 0.2456426390463990
        seed_V[3, 2] = 0.2665038412777220

        clf = GenSVM(
            p=1.2143,
            kappa=0.90298,
            lmd=0.00219038,
            epsilon=1e-15,
            weights="unit",
            kernel="linear",
        )

        clf.fit(X, y, seed_V=seed_V)

        V = clf.combined_coef_
        eps = 1e-7
        self.assertLess(abs(V[0, 0] - -1.1907736868272805), eps)
        self.assertLess(abs(V[0, 1] - 1.8651287814979396), eps)
        self.assertLess(abs(V[0, 2] - 1.7250030581662932), eps)
        self.assertLess(abs(V[1, 0] - 0.7925100058806183), eps)
        self.assertLess(abs(V[1, 1] - -3.6093428916761665), eps)
        self.assertLess(abs(V[1, 2] - -1.3394018960329377), eps)
        self.assertLess(abs(V[2, 0] - 1.5203132433193016), eps)
        self.assertLess(abs(V[2, 1] - -1.9118604362643852), eps)
        self.assertLess(abs(V[2, 2] - -1.7939246097629342), eps)
        self.assertLess(abs(V[3, 0] - 0.0658817457370326), eps)
        self.assertLess(abs(V[3, 1] - 0.6547924025329720), eps)
        self.assertLess(abs(V[3, 2] - -0.6773346708737853), eps)

    def test_fit_with_weights(self):
        """ GENSVM: Test fit with sample weights """
        X, y = load_iris(return_X_y=True)
        weights = np.random.random((X.shape[0],))
        clf = GenSVM(max_iter=100)
        clf.fit(X, y, sample_weight=weights)
        # with seeding
        V = clf.combined_coef_
        weights = np.random.random((X.shape[0],))
        clf.fit(X, y, sample_weight=weights, seed_V=V)

        clf = GenSVM(kernel="rbf")
        clf.fit(X, y, sample_weight=weights)


if __name__ == "__main__":
    unittest.main()
