import numpy as np

import util

import unittest

class TestUtil(unittest.TestCase) :

    def test_pick_item(self) :
        list_items = np.random.randint(1, 10, size=10)
        sum_items = sum(list_items)
        list_prob = [i/sum_items for i in list_items]
        item = util.pick_item(list_prob)
        self.assertLess(item, len(list_prob))

if __name__ == '__main__' :
    unittest.main()