# -*- coding: utf-8 -*-

"""

    Unit tests for the util module

"""

from __future__ import division, print_function

import numpy as np
import unittest

from gensvm.util import get_ranks


class GenSVMUtilTestCase(unittest.TestCase):
    def test_get_ranks(self):
        """ UTIL: Test ranking function """
        x = [7, 0.1, 0.5, 0.1, 10, 100, 200]
        self.assertEqual(get_ranks(x), [4, 1, 3, 1, 5, 6, 7])

        x = [3, 3, 3]
        self.assertEqual(get_ranks(x), [1, 1, 1])

        x = [1, 2, 3]
        self.assertEqual(get_ranks(x), [1, 2, 3])

        x = [-1, -2, -3]
        self.assertEqual(get_ranks(x), [3, 2, 1])

    def test_get_ranks_nan(self):
        """ UTIL: Test ranking function with NaN entries """
        x = [3, 2, 1, 4, 5, np.nan]
        self.assertEqual(get_ranks(x), [3, 2, 1, 4, 5, 6])

        x = [3, 2, 1, np.nan, 4, 5]
        self.assertEqual(get_ranks(x), [3, 2, 1, 6, 4, 5])

        x = [3, 2, 1, np.nan, 4, 5, np.nan]
        self.assertEqual(get_ranks(x), [3, 2, 1, 6, 4, 5, 6])
