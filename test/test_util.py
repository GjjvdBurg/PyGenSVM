# -*- coding: utf-8 -*-

"""

    Unit tests for the util module

"""

from __future__ import division, print_function

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
