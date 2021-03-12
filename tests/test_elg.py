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
        empty_vals = [
            [],
            [[], [], []]
        ]

        for val in empty_vals :
            with self.subTest(assoc_matrix=val) :
                self.agent.set_assoc_matrix(val)
                self.agent.update_active_matrix()
                self.assertEqual(self.agent.active_matrix.size, 0)
                self.assertEqual(np.shape(self.agent.active_matrix), np.shape(val))

    def test_update_active_matrix_small(self) :
        assoc_matrix = [
            [3, 1],
            [2, 8]
        ]
        expected_active_matrix = np.array([
            [0.75, 0.25],
            [0.2, 0.8]
        ])

        self.agent.set_assoc_matrix(assoc_matrix)
        self.agent.update_active_matrix()
        self.assertTrue(np.array_equal(self.agent.active_matrix, expected_active_matrix))

    def test_update_active_matrix_large(self) :
        assoc_matrix = [
            [3, 1, 5, 6, 1, 4, 5],
            [2, 7, 5, 1, 9, 0, 1],
            [7, 4, 2, 0, 3, 7, 2],
            [1, 0, 4, 5, 6, 3, 6]
        ]
        expected_active_matrix = np.array([
            [0.12, 0.04, 0.2, 0.24, 0.04, 0.16, 0.2],
            [0.08, 0.28, 0.2, 0.04, 0.36, 0, 0.04],
            [0.28, 0.16, 0.08, 0, 0.12, 0.28, 0.08],
            [0.04, 0, 0.16, 0.2, 0.24, 0.12, 0.24]
        ])

        self.agent.set_assoc_matrix(assoc_matrix)
        self.agent.update_active_matrix()
        self.assertTrue(np.array_equal(self.agent.active_matrix, expected_active_matrix))

    def test_update_active_matrix_sum(self) :
        for i in range(10) :
            s = tuple(np.random.randint(100, size=2))
            with self.subTest(shape=s) :
                assoc_matrix = src.elg.random_assoc_matrix(*s)
                self.agent.set_assoc_matrix(assoc_matrix)
                self.agent.update_active_matrix()
                self.assertEqual(np.shape(self.agent.active_matrix), s)
                for row in self.agent.active_matrix :
                    with self.subTest() :
                        self.assertAlmostEqual(sum(row), 1, places=5)

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