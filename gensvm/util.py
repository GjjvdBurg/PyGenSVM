# -*- coding: utf-8 -*-

"""
Utility functions for GenSVM

"""


import numpy as np


def get_ranks(x):
    """
    Rank data in an array. Low values get a small rank number. Ties are broken 
    by assigning the lowest value.

    Examples
    --------
    >>> x = [7, 0.1, 0.5, 0.1, 10, 100, 200]
    >>> get_ranks(x)
    [4, 1, 3, 1, 5, 6, 7]

    """
    x = np.ravel(np.asarray(x))
    l = len(x)
    r = 1
    ranks = np.zeros((l, ))
    while not all([k is None for k in x]):
        m = min([k for k in x if not k is None])
        idx = [1 if k == m else 0 for k in x]
        ranks = [r if idx[k] else ranks[k] for k in range(l)]
        r += sum(idx)
        x = [None if idx[k] else x[k] for k in range(l)]
    return ranks
