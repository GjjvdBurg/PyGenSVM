#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for evaluating interrupted training.

This test requires human interaction.

Author: Gertjan van den Burg

"""

from gensvm import GenSVM, GenSVMGridSearchCV
from sklearn.datasets import load_iris


def wait_for_human():
    print("\nThe goal of this test is to interrupt training using Ctrl+C.")
    input("Are you ready to interrupt training? Press enter to continue.")


def single():
    wait_for_human()
    X, y = load_iris(return_X_y=True)
    clf = GenSVM(lmd=1e-10, epsilon=1e-10, p=2, kappa=-0.99, verbose=1)
    clf.fit(X, y)


def grid():
    wait_for_human()
    X, y = load_iris(return_X_y=True)
    gg = GenSVMGridSearchCV(
        {"p": [1.0, 2.0], "kappa": [1, 2], "lmd": [1e-5]}, verbose=1
    )
    gg.fit(X, y)


def main():
    single()
    grid()


if __name__ == "__main__":
    main()
