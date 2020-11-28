from unittest import TestCase

tc = TestCase()


def assert_matrices_equal(expected, actual):
    for (index, val) in enumerate(['TP', 'FP', 'FN', 'TN']):
        expected_val = expected[index // 2][index % 2]
        actual_val = actual[index // 2][index % 2]
        tc.assertAlmostEqual(expected_val, actual_val, delta = 1, msg = f'{val} should be {expected_val} but was {actual_val}')
