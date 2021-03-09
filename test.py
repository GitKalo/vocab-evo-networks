import numpy as np

import util

import unittest

class TestUtil(unittest.TestCase) :
    def test_pick_item_success(self) :
        list_items = np.random.randint(1, 10, size=10)
        sum_items = sum(list_items)
        list_prob = [i/sum_items for i in list_items]
        item = util.pick_item(list_prob)
        self.assertLess(item, len(list_prob), 'Returned index is out of range for the list provided.')

    def test_pick_item_wrong_list_distribution(self) :
        list_items = np.random.randint(1, 10, size=10)
        self.assertRaises(AssertionError, util.pick_item, list_items)
    
    def test_pick_item_wrong_list_values(self) :
        self.assertRaises(ValueError, util.pick_item, [])
        self.assertRaises(ValueError, util.pick_item, ['test'])
        self.assertRaises(ValueError, util.pick_item, [1, 'test', 0.2])

        try :
            input_value = [1, 0.2]
            util.pick_item(input_value)
        except ValueError :
            self.fail("pick_item raised unexpected ValueError on input " + str(input_value))

    def test_pick_item_wrong_arg_type(self) :
        self.assertRaises(TypeError, util.pick_item, 5)
        self.assertRaises(TypeError, util.pick_item, 'test')
        self.assertRaises(TypeError, util.pick_item, (1, ['a', 3]))

if __name__ == '__main__' :
    unittest.main()