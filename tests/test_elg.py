import numpy as np

import unittest

from context import src

class TestELG(unittest.TestCase) :
    def setUp(self) :
        self.agent = src.elg.Agent(0)

    def test_update_language_empty(self) :
        empty_vals = [None, []]
        for val in empty_vals :
            with self.subTest(assoc_matrix=val) :
                self.agent.update_language(val)
                self.assertEqual(self.agent.assoc_matrix, [])
                self.assertEqual(self.agent.active_matrix.size, 0)
                self.assertEqual(self.agent.passive_matrix.size, 0)

    def test_update_language_shape(self) :
        shapes = [
            (1, 1),
            (5, 5),
            (100, 100),
            (1, 10),
            (5, 100),
            (10, 1),
            (100, 5)
        ]

        for shape in shapes :
            with self.subTest(assoc_matrix_shape=shape) :
                self.agent.update_language(np.zeros(shape))
                self.assertEqual(np.shape(self.agent.active_matrix), shape)
                self.assertEqual(np.shape(self.agent.passive_matrix), shape[::-1])

    def test_update_active_matrix_empty(self) :
        # test empty assoc matrix
        # test empty rows assoc matrix
        pass

    def test_update_active_matrix_small(self) :
        pass

    def test_update_active_matrix_large(self) :
        pass

    def test_update_passive_matrix_empty(self) :
        # test empty assoc matrix
        # test empty rows assoc matrix
        pass

    def test_update_passive_matrix_small(self) :
        pass

    def test_update_passive_matrix_large(self) :
        pass

    def test_random_assoc_matrix_shape(self) :
        pass

    def test_payoff_small(self) :
        pass

    def test_payoff_large(self) :
        pass
    
    def test_payoff_symmetry(self) :
        pass

    def test_sample_none(self) :
        pass

    def test_sample_empty(self) :
        pass

    def test_sample_failed_pick_item(self) :
        pass

    def test_sample_small_square(self) :
        pass

    def test_sample_small_rect(self) :
        pass

    def test_sample_large_square(self) :
        pass

    def test_sample_large_rect(self) :
        pass

if __name__ == '__main__' :
    unittest.main()