import numpy as np

import unittest

from context import src

class TestAgent(unittest.TestCase) :
    def setUp(self) :
        self.agent = src.agent.Agent(0)

    def test_update_language_empty(self) :
        self.agent.update_language([[]])
        self.assertEqual(self.agent.assoc_matrix.size, 0)
        self.assertEqual(self.agent.active_matrix.size, 0)
        self.assertEqual(self.agent.passive_matrix.size, 0)

    def test_update_language_malformed(self) :
        vals = [None, []]
        for val in vals :
            with self.subTest(assoc_matrix=val) :
                self.assertRaises(ValueError, self.agent.update_language, val)

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
        self.agent.update_language([[]])
        self.assertEqual(self.agent.active_matrix.size, 0)
        self.assertEqual(np.shape(self.agent.active_matrix), np.shape([[]]))

    def test_update_active_matrix_small(self) :
        assoc_matrix = [
            [3, 1],
            [2, 8]
        ]
        expected_active_matrix = np.array([
            [0.75, 0.25],
            [0.2, 0.8]
        ])

        self.agent.update_language(assoc_matrix)
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

        self.agent.update_language(assoc_matrix)
        self.assertTrue(np.array_equal(self.agent.active_matrix, expected_active_matrix))

    def test_update_active_matrix_sum(self) :
        for i in range(10) :
            s = tuple(np.random.randint(1, 100, size=2))
            with self.subTest(assoc_matrix_shape=s) :
                assoc_matrix = src.agent.random_assoc_matrix(*s)
                self.agent.update_language(assoc_matrix)
                self.assertEqual(np.shape(self.agent.active_matrix), s)
                for row in self.agent.active_matrix :
                    with self.subTest(row=row) :
                        self.assertAlmostEqual(sum(row), 1, places=5)

    def test_update_passive_matrix_empty(self) :
        self.agent.update_language([[]])
        self.assertEqual(self.agent.passive_matrix.size, 0)
        self.assertEqual(np.shape(self.agent.passive_matrix), np.shape([[]])[::-1])

    def test_update_passive_matrix_small(self) :
        assoc_matrix = [
            [6, 3],
            [4, 1]
        ]
        expected_passive_matrix = np.array([
            [0.6, 0.4],
            [0.75, 0.25]
        ])

        self.agent.update_language(assoc_matrix)
        self.assertTrue(np.array_equal(self.agent.passive_matrix, expected_passive_matrix))

    def test_update_passive_matrix_large(self) :
        assoc_matrix = [
            [4, 0, 5, 8, 1, 5, 5],
            [3, 7, 5, 3, 9, 0, 1],
            [8, 6, 2, 0, 2, 7, 3],
            [1, 3, 4, 5, 4, 4, 7]
        ]
        expected_passive_matrix = np.array([
            [0.25, 0.1875, 0.5, 0.0625],
            [0, 0.4375, 0.375, 0.1875],
            [0.3125, 0.3125, 0.125, 0.25],
            [0.5, 0.1875, 0, 0.3125],
            [0.0625, 0.5625, 0.125, 0.25],
            [0.3125, 0, 0.4375, 0.25],
            [0.3125, 0.0625, 0.1875, 0.4375]
        ])

        self.agent.update_language(assoc_matrix)
        self.assertTrue(np.array_equal(self.agent.passive_matrix, expected_passive_matrix))

    def test_update_passive_matrix_sum(self) :
        for i in range(10) :
            s = tuple(np.random.randint(1, 100, size=2))
            with self.subTest(assoc_matrix_shape=s) :
                assoc_matrix = src.agent.random_assoc_matrix(*s)
                self.agent.update_language(assoc_matrix)
                self.assertEqual(np.shape(self.agent.passive_matrix), s[::-1])
                for row in self.agent.passive_matrix :
                    with self.subTest(row=row) :
                        self.assertAlmostEqual(sum(row), 1, places=5)

    def test_random_assoc_matrix_shape(self) :
        s = tuple(np.random.randint(1, 100, size=2))
        self.assertEqual(s, np.shape(src.agent.random_assoc_matrix(*s)))

    def test_payoff_different_shapes(self) :
        agent_1 = src.agent.Agent(1)
        agent_2 = src.agent.Agent(2)

        agent_1.update_language(src.agent.random_assoc_matrix(*np.random.randint(1, 50, size=2)))
        agent_2.update_language(src.agent.random_assoc_matrix(*np.random.randint(50, 100, size=2)))

        self.assertRaises(ValueError, src.agent.payoff, agent_1, agent_2)

    def test_payoff_small(self) :
        agent_1 = src.agent.Agent(1)
        agent_2 = src.agent.Agent(2)

        agent_1.update_language([
            [6, 6],
            [4, 8]
        ])
        agent_2.update_language([
            [2, 6],
            [4, 4]
        ])

        payoff = src.agent.payoff(agent_1, agent_2)
        self.assertAlmostEqual(payoff, 0.95635, 5)

    def test_payoff_large(self) :
        agent_1 = src.agent.Agent(1)
        agent_2 = src.agent.Agent(2)

        agent_1.update_language([
            [6, 2, 0, 2],
            [3, 4, 1, 2],
            [2, 3, 4, 1],
            [4, 1, 5, 0],
            [0, 3, 2, 5]
        ])
        agent_2.update_language([
            [3, 2, 0, 5],
            [1, 7, 2, 0],
            [4, 0, 2, 4],
            [0, 2, 8, 0],
            [5, 1, 3, 1]
        ])

        payoff = src.agent.payoff(agent_1, agent_2)
        self.assertAlmostEqual(payoff, 1.11467, 4)
    
    def test_payoff_symmetry(self) :
        for i in range(10) :
            agent_1 = src.agent.Agent(1)
            agent_2 = src.agent.Agent(2)

            agent_1.update_language(src.agent.random_assoc_matrix(5, 5))
            agent_2.update_language(src.agent.random_assoc_matrix(5, 5))

            payoff_1_2 = src.agent.payoff(agent_1, agent_2)
            payoff_2_1 = src.agent.payoff(agent_2, agent_1)

            with self.subTest(p1=payoff_1_2, p2=payoff_2_1) :
                self.assertEqual(payoff_1_2, payoff_2_1)

    def test_sample_none(self) :
        sample = src.agent.sample(self.agent, 0)
        self.assertEqual(sample.sum(), 0)

    def test_sample_empty(self) :
        self.agent.update_language([[]])
        self.assertEqual(src.agent.sample(self.agent, 1).size, 0)

    def test_sample_number(self) :
        for i in range(10) :
            k = np.random.randint(100)
            with self.subTest(k=k) :
                self.agent.update_language(src.agent.random_assoc_matrix(*np.random.randint(1, 100, size=2)))
                sample = src.agent.sample(self.agent, k)
                for row in sample :
                    with self.subTest(row=row) :
                        self.assertEqual(sum(row), k)

    @unittest.skip("Not implemented")
    def test_sample_method_equivalence(self) :
        pass

if __name__ == '__main__' :
    unittest.main(verbosity=2)