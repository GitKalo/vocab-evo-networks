import numpy as np

import unittest

from context import src

class TestELG(unittest.TestCase) :
    def setUp(self) :
        self.agent = src.elg.Agent(0)

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