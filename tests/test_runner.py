import unittest, os

import test_util, test_elg

tests = unittest.TestLoader().discover(os.getcwd())

suite = unittest.TestSuite(tests)

unittest.TextTestRunner(verbosity=2).run(suite)