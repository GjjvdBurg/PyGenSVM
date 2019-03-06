# -*- coding: utf-8 -*-

"""
Utility functions for GenSVM

"""


import numpy as np


def get_ranks(a):
    """
    Rank data in an array. Low values get a small rank number. Ties are broken 
    by assigning the lowest value (this corresponds to ``rankdata(a, 
    method='min')`` in SciPy.

    Examples
    --------
    >>> x = [7, 0.1, 0.5, 0.1, 10, 100, 200]
    >>> get_ranks(x)
    [4, 1, 3, 1, 5, 6, 7]

    """
    orig = np.ravel(np.asarray(a))
    arr = orig[~np.isnan(orig)]
    sorter = np.argsort(arr, kind="quicksort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    count = np.r_[np.nonzero(obs)[0], len(obs)]
    ranks = np.zeros_like(orig)
    ranks[~np.isnan(orig)] = count[dense - 1] + 1
    ranks[np.isnan(orig)] = np.max(ranks) + 1
    return list(ranks)
