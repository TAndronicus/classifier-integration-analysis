import unittest

from etl import Etl
from etl_test_helper import EtlTestHelper


class EtlTest(unittest.TestCase):
    etl = Etl()
    etlHelper = EtlTestHelper()

    def test1(self):
        self.etlHelper.assert_matrices_equal(
            [
                [50, 50],
                [0, 0]
            ],
            self.etl.calculate_conf_matrix(.5, .5, 1, 100)
        )

    def test2(self):
        self.etlHelper.assert_matrices_equal(
            [
                [50, 25],
                [25, 0]
            ],
            self.etl.calculate_conf_matrix(.5, 2 / 3, 2 / 3, 100)
        )

    def test3(self):
        self.etlHelper.assert_matrices_equal(
            [
                [450, 300],
                [200, 700]
            ],
            self.etl.calculate_conf_matrix(.6970, .6, 9 / 13, 1650)
        )

    def test4(self):
        self.etlHelper.assert_matrices_equal(
            [
                [50, 0],
                [50, 0]
            ],
            self.etl.calculate_conf_matrix(.5, 1, .5, 100)
        )

    def test5(self):
        matrix = self.etl.calculate_conf_matrix(.8, .8, 1, 100)
        self.etl.print_conf_matrix(matrix)
        self.etlHelper.assert_matrices_equal(
            [
                [80, 20],
                [0, 0]
            ],
            matrix
        )

    def test6(self):
        self.etlHelper.assert_matrices_equal(
            [
                [80, 10],
                [10, 0]
            ],
            self.etl.calculate_conf_matrix(.8, 8 / 9, 8 / 9, 100)
        )

    def test7(self):
        self.etlHelper.assert_matrices_equal(
            [
                [80, 0],
                [0, 20]
            ],
            self.etl.calculate_conf_matrix(1, 1, 1, 100)
        )

    def test8(self):
        self.etlHelper.assert_matrices_equal(
            [
                [0, 70],
                [30, 0]
            ],
            self.etl.calculate_conf_matrix(0, 0, 0, 100)
        )

    def test9(self):
        self.etlHelper.assert_matrices_equal(
            [
                [100, 0],
                [0, 0]
            ],
            self.etl.calculate_conf_matrix(1, 1, 1, 100)
        )

    def test10(self):
        self.etlHelper.assert_matrices_equal(
            [
                [0, 0],
                [100, 0]
            ],
            self.etl.calculate_conf_matrix(0, 0, 0, 100)
        )

    def test11(self):
        self.etlHelper.assert_matrices_equal(
            [
                [23, 17],
                [13, 19]
            ],
            self.etl.calculate_conf_matrix(.5833, 0.575, 0.6389, 72)
        )

    def test12(self):
        self.etlHelper.assert_matrices_equal(
            [
                [97, 1],
                [2, 0]
            ],
            self.etl.calculate_conf_matrix(.97, .9898, .9798, 100)
        )

    def test13(self):
        self.etlHelper.assert_matrices_equal(
            [
                [94, 1],
                [2, 3]
            ],
            self.etl.calculate_conf_matrix(.97, 0.9895, .9792, 100)
        )

    def test14(self):
        self.etlHelper.assert_matrices_equal(
            [
                [4500, 3000],
                [2000, 7000]
            ],
            self.etl.calculate_conf_matrix(.697, .6, 9 / 13, 16500)
        )

    def test15(self):
        self.etlHelper.assert_matrices_equal(
            [
                [0, 30],
                [0, 70]
            ],
            self.etl.calculate_conf_matrix(.7, 0, 0, 100)
        )


if __name__ == '__main__':
    unittest.main()
