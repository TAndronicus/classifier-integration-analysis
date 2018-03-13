import unittest
import ClassifLibrary
import numpy as np
import math
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from ClassifLibrary import ClassifierData
from MergingAlgorithm import apply


class MyLibraryTest(unittest.TestCase):
    X, y = [], []
    NUMBER_OF_CLASSIFIERS = 3
    NUMBER_OF_SUBSPACES = 5
    NUMBER_OF_ATTRIBUTES = 2
    TEST_FILENAME = 'Dane_9_12_2017.xlsx'
    QUOTIENT = 2 / 3
    conf_matrix = [[400, 100], [200, 300]]

    def setUp(self):
        self.X, self.y = \
            make_classification(n_features = 2, n_informative = 2, n_redundant = 0, n_repeated = 0, n_samples = 1000,
                                class_sep = 2.7, hypercube = False, random_state = 2, n_clusters_per_class = 1)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_should_recognise_linear_classifier(self):
        # given
        clf = LinearSVC()
        clf.fit(self.X, self.y)
        # when
        clf_type = ClassifLibrary.determine_clf_type(clf)
        # then
        self.assertEqual(clf_type, ClassifLibrary.ClfType.LINEAR)

    def test_should_recognise_mean_classifier(self):
        # given
        clf = NearestCentroid()
        clf.fit(self.X, self.y)
        # when
        clf_type = ClassifLibrary.determine_clf_type(clf)
        # then
        self.assertEqual(clf_type, ClassifLibrary.ClfType.MEAN)

    def ignore_test_should_not_recognise_not_trained_classifier(self):
        # given
        clf = LinearSVC()
        # when
        ClassifLibrary.determine_clf_type(clf)
        # then
        self.assertRaises(Exception('Classifier not defined'))

    def test_should_initialse_right_number_of_classifiers(self):
        # given
        # when
        clfs = ClassifLibrary.initialize_classifiers()
        # then
        self.assertEqual(len(clfs), self.NUMBER_OF_CLASSIFIERS)

    def test_should_return_non_empty_dataset(self):
        # given
        # when
        X, y = ClassifLibrary.prepare_raw_data()
        # then
        self.assertFalse(len(X) == 0)

    def test_should_return_class_for_each_data(self):
        # given
        # when
        X, y = ClassifLibrary.prepare_raw_data()
        # then
        self.assertTrue(len(X) == len(y))

    def test_should_return_sorted_data(self):
        # given
        # when
        X, y = ClassifLibrary.prepare_raw_data()
        # then
        for i in range(len(X) - 1):
            self.assertFalse((X[i + 1][0] >= X[i][0]) ^ (y[i + 1] == y[i]))

    def test_should_return_sorted_data_from_dataset(self):
        # given
        # when
        X, y = ClassifLibrary.load_samples_from_datasets_first_two_rows()
        # then
        for i in range(len(X) - 1):
            self.assertFalse((X[i + 1][0] >= X[i][0]) ^ (y[i + 1] == y[i]))

    def test_should_return_dataset_with_two_attributes(self):
        # given
        # when
        X, y = ClassifLibrary.load_samples_from_datasets_first_two_rows()
        # then
        self.assertEqual(self.NUMBER_OF_ATTRIBUTES, np.shape(X)[1])

    def test_should_not_change_data(self):
        # given
        # when
        X1, y1 = ClassifLibrary.load_samples_from_file_non_parametrized(self.TEST_FILENAME)
        X2, y2 = ClassifLibrary.load_samples_from_datasets_first_two_rows()
        # then
        self.assertTrue(len(X2) <= len(X1))
        for i in range(len(X2)):
            self.assertTrue(X1.__contains__(X2[i]))

    def test_should_not_change_data_whole(self):
        # given
        data = ClassifierData(are_samples_generated = False, filename = 'datasets.xlsx')
        # when
        X1, y1 = ClassifLibrary.load_samples_from_file_non_parametrized(self.TEST_FILENAME)
        X2, y2 = ClassifLibrary.prepare_raw_data(data)
        # then
        self.assertTrue(len(X2) <= len(X1))
        self.assertEqual(len(X1[0]), 2)
        self.assertEqual(len(X2[0]), 2)

    def test_should_contain_same_data(self):
        # given
        data = ClassifierData(are_samples_generated = False, filename = 'datasets.xlsx')
        # when
        X1, y1 = ClassifLibrary.prepare_raw_data(data)
        X2, y2 = ClassifLibrary.load_samples_from_datasets_first_two_rows()
        # then
        self.assertTrue(len(X2) == len(X1))
        self.assertEqual(len(X1[0]), 2)
        self.assertEqual(len(X2[0]), 2)

    def test_should_return_sorted_data_from_dataset_given_columns(self):
        # given
        # when
        X, y = ClassifLibrary.load_samples_from_datasets_non_parametrised()
        # then
        for i in range(len(X) - 1):
            self.assertFalse((X[i + 1][0] >= X[i][0]) ^ (y[i + 1] == y[i]))

    def test_should_return_dataset_with_two_attributes_given_columns(self):
        # given
        # when
        X, y = ClassifLibrary.load_samples_from_datasets_non_parametrised()
        # then
        self.assertEqual(self.NUMBER_OF_ATTRIBUTES, np.shape(X)[1])

    def test_should_not_change_data_given_columns(self):
        # given
        # when
        X1, y1 = ClassifLibrary.load_samples_from_file_non_parametrized(self.TEST_FILENAME)
        X2, y2 = ClassifLibrary.load_samples_from_datasets_non_parametrised()
        # then
        self.assertTrue(len(X2) <= len(X1))
        for i in range(len(X2)):
            self.assertTrue(X1.__contains__(X2[i]))

    def test_should_contain_same_data_given_columns(self):
        # given
        data = ClassifierData(are_samples_generated = False, filename = 'datasets.xlsx')
        # when
        X1, y1 = ClassifLibrary.prepare_raw_data(data)
        X2, y2 = ClassifLibrary.load_samples_from_datasets_non_parametrised()
        # then
        self.assertTrue(len(X2) == len(X1))
        self.assertEqual(len(X1[0]), 2)
        self.assertEqual(len(X2[0]), 2)

    def test_cumulative_length_of_returned_datasets_should_be_multiply_of_number_of_subspaces(self):
        # given
        X0_full = np.array(
            [[0, 0], [0, 0], [0, 0], [21, 0], [21, 0], [42, 0], [42, 0], [42, 0], [63, 0], [63, 0], [84, 0], [84, 0],
             [84, 0], [100, 0], [100, 0]])
        X1_full = np.array(
            [[0, 0], [0, 0], [21, 0], [21, 0], [21, 0], [42, 0], [42, 0], [63, 0], [63, 0], [63, 0], [84, 0], [84, 0],
             [100, 0], [100, 0], [100, 0]])
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_full, X1_full)
        # then
        self.assertTrue((len(X0) + len(X1)) % (self.NUMBER_OF_CLASSIFIERS + 2) == 0)

    def test_should_return_datasets_of_same_length(self):
        # given
        X0_full = np.array([[0, 0], [0, 0], [0, 0], [21, 0], [21, 0], [42, 0], [42, 0], [42, 0], [63, 0], [63, 0],
                            [84, 0], [84, 0], [84, 0], [100, 0], [100, 0]])
        X1_full = np.array([[0, 0], [0, 0], [21, 0], [21, 0], [21, 0], [42, 0], [42, 0], [63, 0], [63, 0], [63, 0],
                            [84, 0], [84, 0], [100, 0], [100, 0], [100, 0]])
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_full, X1_full)
        # then
        self.assertEqual(len(X0_full), len(X0))
        self.assertEqual(len(X1_full), len(X1))

    def test_should_cut_off_redundant_data_at_front(self):
        # given
        X0_full = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [21, 0], [21, 0], [42, 0], [42, 0], [42, 0], [63, 0],
                            [63, 0], [84, 0], [84, 0], [84, 0], [100, 0], [100, 0]])
        X1_full = np.array([[0, 0], [0, 0], [2, 0], [21, 0], [21, 0], [21, 0], [42, 0], [42, 0], [63, 0], [63, 0],
                            [63, 0], [84, 0], [84, 0], [100, 0], [100, 0], [100, 0]])
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_full, X1_full)
        # then
        self.assertEqual(len(X0_full) - 1, len(X0))
        self.assertEqual(len(X1_full) - 1, len(X1))

    def test_should_cut_off_redundant_data_at_end(self):
        # given
        X0_full = np.array([[0, 0], [0, 0], [0, 0], [21, 0], [21, 0], [42, 0], [42, 0], [42, 0], [63, 0], [63, 0],
                            [84, 0], [84, 0], [84, 0], [90, 0], [95, 0], [100, 0], [100, 0]])
        X1_full = np.array([[0, 0], [0, 0], [21, 0], [21, 0], [21, 0], [42, 0], [42, 0], [63, 0], [63, 0], [63, 0],
                            [84, 0], [84, 0], [90, 0], [100, 0], [100, 0], [100, 0]])
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_full, X1_full)
        # then
        self.assertEqual(len(X0_full) - 2, len(X0))
        self.assertEqual(len(X1_full) - 1, len(X1))

    def test_should_cut_off_redundant_data_in_middle(self):
        # given
        X0_full = np.array([[0, 0], [0, 0], [0, 0], [21, 0], [21, 0], [25, 0], [42, 0], [42, 0], [42, 0], [63, 0],
                            [63, 0], [84, 0], [84, 0], [84, 0], [100, 0], [100, 0]])
        X1_full = np.array([[0, 0], [0, 0], [21, 0], [21, 0], [21, 0], [42, 0], [42, 0], [45, 0], [63, 0], [63, 0],
                            [63, 0], [84, 0], [84, 0], [85, 0], [100, 0], [100, 0], [100, 0]])
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_full, X1_full)
        # then
        self.assertEqual(len(X0_full) - 1, len(X0))
        self.assertEqual(len(X1_full) - 2, len(X1))

    def test_should_have_amount_of_data_multiple_of_number_of_classifiers_plus_2_in_every_subspace_own_dataset(self):
        # given
        X0_full = np.array(
            [[0, 0], [0, 0], [0, 0], [21, 0], [21, 0], [25, 0], [42, 0], [42, 0], [42, 0], [63, 0], [63, 0], [84, 0],
             [84, 0], [84, 0], [100, 0], [100, 0]])
        X1_full = np.array(
            [[0, 0], [0, 0], [21, 0], [21, 0], [21, 0], [42, 0], [42, 0], [45, 0], [63, 0], [63, 0], [63, 0], [84, 0],
             [84, 0], [85, 0], [100, 0], [100, 0],
             [100, 0]])
        lengths0, lengths1 = [], []
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_full, X1_full)
        min_x0, max_x0, min_x1, max_x1 = X0[0][0], X0[0][-1], X1[0][0], X1[0][-1]
        for i in range(self.NUMBER_OF_SUBSPACES):
            amount = 0
            for j in range(len(X0)):
                if min_x0 + i * (max_x0 - min_x0) / self.NUMBER_OF_SUBSPACES <= X0[j][0] <= \
                        min_x0 + (i + 1) * (max_x0 - min_x0) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths0.append(amount)
            amount = 0
            for j in range(len(X1)):
                if min_x1 + i * (max_x1 - min_x1) / self.NUMBER_OF_SUBSPACES <= X1[j][0] <= \
                        min_x1 + (i + 1) * (max_x1 - min_x1) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths1.append(amount)
        # then
        for i in range(self.NUMBER_OF_SUBSPACES):
            self.assertEqual(0, (lengths0[i] + lengths1[i]) % (self.NUMBER_OF_CLASSIFIERS + 2))

    def test_should_have_amount_of_data_multiple_of_number_of_classifiers_plus_2_in_every_subspace_generated_dataset(
            self):
        # given
        X0_raw, X1_raw = ClassifLibrary.divide_generated_samples(self.X, self.y)
        X0_sorted, X1_sorted = ClassifLibrary.sort_attributes(X0_raw), ClassifLibrary.sort_attributes(X1_raw)
        lengths0, lengths1 = [], []
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_sorted, X1_sorted)
        length0, length1 = len(X0), len(X1)
        print(len(X0))
        while True:
            X0, X1 = ClassifLibrary.assert_distribution(X0, X1)
            if len(X0) == length0 and len(X1) == length1:
                break
            length0, length1 = len(X0), len(X1)
        min_x0, max_x0, min_x1, max_x1 = X0[0][0], X0[-1][0], X1[0][0], X1[-1][0]
        min_x, max_x = min(min_x0, min_x1), max(max_x0, max_x1)
        for i in range(self.NUMBER_OF_SUBSPACES):
            amount = 0
            for j in range(len(X0)):
                if min_x + i * (max_x - min_x) / self.NUMBER_OF_SUBSPACES <= X0[j][0] <= \
                        min_x + (i + 1) * (max_x - min_x) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths0.append(amount)
            amount = 0
            for j in range(len(X1)):
                if min_x + i * (max_x - min_x) / self.NUMBER_OF_SUBSPACES <= X1[j][0] <= \
                        min_x + (i + 1) * (max_x - min_x) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths1.append(amount)
        # then
        for i in range(self.NUMBER_OF_SUBSPACES):
            self.assertEqual(0, (lengths0[i] + lengths1[i]) % (self.NUMBER_OF_CLASSIFIERS + 2))

    def test_should_have_correct_amount_of_data_in_every_subspace_generated_dataset_simplified(self):
        # given
        X0_raw, X1_raw = ClassifLibrary.divide_generated_samples(self.X, self.y)
        X0_sorted, X1_sorted = ClassifLibrary.sort_attributes(X0_raw), ClassifLibrary.sort_attributes(X1_raw)
        lengths0, lengths1 = [], []
        # when
        X0, X1 = ClassifLibrary.assert_distribution_simplified(X0_sorted, X1_sorted)
        X0, X1 = ClassifLibrary.assert_distribution_simplified(X0, X1)
        min_x0, max_x0, min_x1, max_x1 = X0[0][0], X0[-1][0], X1[0][0], X1[-1][0]
        min_x, max_x = min(min_x0, min_x1), max(max_x0, max_x1)
        for i in range(self.NUMBER_OF_SUBSPACES):
            amount = 0
            for j in range(len(X0)):
                if min_x + i * (max_x - min_x) / self.NUMBER_OF_SUBSPACES <= X0[j][0] <= \
                        min_x + (i + 1) * (max_x - min_x) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths0.append(amount)
            amount = 0
            for j in range(len(X1)):
                if min_x + i * (max_x - min_x) / self.NUMBER_OF_SUBSPACES <= X1[j][0] <= \
                        min_x + (i + 1) * (max_x - min_x) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths1.append(amount)
        # then
        for i in range(self.NUMBER_OF_SUBSPACES):
            self.assertEqual(0, (lengths0[i] + lengths1[i]) % (self.NUMBER_OF_CLASSIFIERS + 2))

    def test_should_return_right_extrema(self):
        # given
        X0 = np.array([[-20, 0], [0, 0], [10, 0]])
        X1 = np.array([[-10, 0], [0, 0], [20, 0]])
        # when
        x_min, x_max = ClassifLibrary.get_extrema_for_subspaces(X0, X1)
        # then
        self.assertEqual(X0[0][0], x_min)
        self.assertEqual(X1[-1][0], x_max)

    def test_should_return_right_count_and_index(self):
        # given
        X = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [3, 0], [10, 0]])
        # when
        counter, index = \
            ClassifLibrary.get_count_of_samples_in_subspace_and_beginning_index_of_next_subspace(X, X[0][0],
                                                                                                 X[-1][0], 1)
        # then
        self.assertEqual(1, counter)
        self.assertEqual(5, index)

    def test_should_return_right_subtraction_and_rest_when_counter_bigger(self):
        # given
        counter, remainder = 5, 3
        # when
        subtraction, rest = ClassifLibrary.set_subtraction_and_rest(counter, remainder)
        # then
        self.assertEqual(remainder, subtraction)
        self.assertEqual(0, rest)

    def test_should_return_right_subtraction_and_rest_when_counter_smaller(self):
        # given
        counter, remainder = 3, 5
        # when
        subtraction, rest = ClassifLibrary.set_subtraction_and_rest(counter, remainder)
        # then
        self.assertEqual(counter, subtraction)
        self.assertEqual(remainder - subtraction, rest)

    def test_should_limit_datasets_for_every_subspace_but_last(self):
        # given
        X0_raw = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0],
                           [12, 0], [13, 0], [14, 0], [15, 0], [16, 0]])
        X1_raw = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0],
                           [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]])
        counter, remainder, index0, index1, is_first_bigger = 2, 3, int(len(X0_raw) / 2), int(len(X1_raw) / 2), False
        # when
        X0, X1 = \
            ClassifLibrary.limit_datasets_for_every_subspace_but_last(X0_raw, X1_raw, counter, remainder,
                                                                      index0, index1, is_first_bigger)
        # then
        self.assertEqual(remainder - counter, len(X0_raw) - len(X0))
        self.assertEqual(counter, len(X1_raw) - len(X1))

    def test_should_limit_datasets_for_last_subspace(self):
        # given
        X0_raw = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0],
                           [12, 0], [13, 0], [14, 0], [15, 0], [16, 0]])
        X1_raw = np.array([[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0],
                           [12, 0], [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]])
        counter, remainder, index0, index1, is_first_bigger = 2, 3, int(len(X0_raw) / 2), int(len(X1_raw) / 2), False
        # when
        X0, X1 = \
            ClassifLibrary.limit_datasets_for_last_subspace(X0_raw, X1_raw, counter, remainder, index0, index1,
                                                            is_first_bigger)
        # then
        self.assertEqual(remainder - counter, len(X0_raw) - len(X0))
        self.assertEqual(counter, len(X1_raw) - len(X1))

    def test_should_limit_datasets_when_last(self):
        # given
        X0_raw = np.array(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
             [13, 0], [14, 0], [15, 0], [16, 0]])
        X1_raw = np.array(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
             [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]])
        counter, remainder, index0, index1, is_first_bigger, is_last = 2, 3, int(len(X0_raw) / 2), \
                                                                       int(len(X1_raw) / 2), False, True
        # when
        X0, X1 = \
            ClassifLibrary.limit_datasets(X0_raw, X1_raw, counter, remainder, index0, index1, is_first_bigger, is_last)
        # then
        self.assertEqual(remainder - counter, len(X0_raw) - len(X0))
        self.assertEqual(counter, len(X1_raw) - len(X1))

    def test_should_limit_datasets_when_not_last(self):
        # given
        X0_raw = np.array(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
             [13, 0], [14, 0], [15, 0], [16, 0]])
        X1_raw = np.array(
            [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], [12, 0],
             [13, 0], [14, 0], [15, 0], [16, 0], [17, 0]])
        counter, remainder, index0, index1, is_first_bigger, is_last = 2, 3, int(len(X0_raw) / 2), \
                                                                       int(len(X1_raw) / 2), False, False
        # when
        X0, X1 = \
            ClassifLibrary.limit_datasets(X0_raw, X1_raw, counter, remainder, index0, index1, is_first_bigger, is_last)
        # then
        self.assertEqual(remainder - counter, len(X0_raw) - len(X0))
        self.assertEqual(counter, len(X1_raw) - len(X1))

    def test_should_have_amount_of_data_as_multiple_of_number_of_classifiers_plus_2_in_every_subspace_for_real_dataset(
            self):
        # given
        X_raw, y_raw = ClassifLibrary.load_samples_from_file_non_parametrized(self.TEST_FILENAME)
        X0_raw, X1_raw = ClassifLibrary.divide_generated_samples(X_raw, y_raw)
        X0_sorted, X1_sorted = ClassifLibrary.sort_attributes(X0_raw), ClassifLibrary.sort_attributes(X1_raw)
        lengths0, lengths1 = [], []
        # when
        X0, X1 = ClassifLibrary.assert_distribution(X0_sorted, X1_sorted)
        length0, length1 = len(X0), len(X1)
        while True:
            X0, X1 = ClassifLibrary.assert_distribution(X0, X1)
            if len(X0) == length0 and len(X1) == length1:
                break
            length0, length1 = len(X0), len(X1)
        min_x0, max_x0, min_x1, max_x1 = X0[0][0], X0[-1][0], X1[0][0], X1[-1][0]
        min_x, max_x = min(min_x0, min_x1), max(max_x0, max_x1)
        for i in range(self.NUMBER_OF_SUBSPACES):
            amount = 0
            for j in range(len(X0)):
                if min_x + i * (max_x - min_x) / self.NUMBER_OF_SUBSPACES <= X0[j][0] <= \
                        min_x + (i + 1) * (max_x - min_x) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths0.append(amount)
            amount = 0
            for j in range(len(X1)):
                if min_x + i * (max_x - min_x) / self.NUMBER_OF_SUBSPACES <= X1[j][0] <= \
                        min_x + (i + 1) * (max_x - min_x) / self.NUMBER_OF_SUBSPACES:
                    amount += 1
            lengths1.append(amount)
        # then
        for i in range(self.NUMBER_OF_SUBSPACES):
            self.assertEqual(0, (lengths0[i] + lengths1[i]) % (self.NUMBER_OF_CLASSIFIERS + 2))

    def test_should_not_loose_data(self):
        # given
        X0 = np.array([[0, 0], [1, 1], [2, 2]])
        X1 = np.array([[3, 3], [4, 4], [5, 5]])
        # when
        X, y = ClassifLibrary.compose_sorted_parts(X0, X1)
        # then
        self.assertEqual(len(X0) + len(X1), len(X))

    def test_should_classify_X0_as_0_and_X1_as_1(self):
        # given
        X0 = np.array([[0, 0], [1, 1], [2, 2]])
        X1 = np.array([[3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])
        # when
        X, y = ClassifLibrary.compose_sorted_parts(X0, X1)
        # then
        self.assertEqual(len(X0), len(y) - np.sum(y))
        self.assertEqual(len(X1), np.sum(y))

    def test_should_return_sorted_array(self):
        # given
        X = np.array([[3, 3], [2, 2], [5, 5], [4, 4], [1, 1]])
        # when
        result = ClassifLibrary.sort_attributes(X)
        # then
        for i in range(len(result) - 1):
            self.assertTrue(result[i + 1][0] > result[i][0])

    def test_should_sort_only_by_X(self):
        # given
        # when
        X, y = ClassifLibrary.sort_results(self.X, self.y)
        # then
        for i in range(len(y) - 1):
            self.assertTrue(X[i][0] <= X[i + 1][0])

    def test_should_divide_by_y(self):
        # given
        # when
        X0, X1 = ClassifLibrary.divide_generated_samples(self.X, self.y)
        # then
        self.assertEqual(len(self.y), len(X0) + len(X1))
        self.assertEqual(len(self.y) - np.sum(self.y), len(X0))
        self.assertEqual(np.sum(self.y), len(X1))

    def test_should_return_training_samples_for_every_classifier(self):
        # given
        # when
        X_whole, y_whole, X_final_test, y_final_test = ClassifLibrary.divide_samples_between_classifiers(self.X, self.y)
        # then
        self.assertEqual(len(X_whole), self.NUMBER_OF_CLASSIFIERS)

    def test_should_return_sorted_training_samples_for_every_classifier(self):
        # given
        # when
        X_whole, y_whole, X_final_test, y_final_test = \
            ClassifLibrary.split_sorted_samples_between_classifiers(self.X, self.y)
        # then
        self.assertEqual(len(X_whole), self.NUMBER_OF_CLASSIFIERS)

    def ignore_deprecated_test_should_return_sets_with_given_ratio(self):
        # given
        # when
        X_train, X_test, y_train, y_test = ClassifLibrary.divide_samples_between_training_and_testing(self.X, self.y)
        # then
        self.assertAlmostEqual(len(X_train) / (len(X_train) + len(X_test)), self.QUOTIENT)

    def test_should_return_sorted_training_samples_for_every_classifier_without_quotient(self):
        # given
        # when
        X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
            ClassifLibrary.split_sorted_samples(self.X, self.y)
        # then
        self.assertEqual(len(X_whole_train), self.NUMBER_OF_CLASSIFIERS)

    def ignore_deprecated_test_train_test_sorted_split(self):
        # given
        # when
        X_train, X_test, y_train, y_test = ClassifLibrary.train_test_sorted_split(self.X, self.y)
        # then
        self.assertAlmostEqual(len(X_train) / (len(X_train) + len(X_test)), self.QUOTIENT)

    def test_should_prepare_samples_for_subspace(self):
        # given
        X = np.array([[3, 3], [1, 1], [5, 5], [4, 4], [1, 1]])
        y = np.array([0, 1, 0, 1, 0])
        # when
        X_sub, y_sub = ClassifLibrary.prepare_samples_for_subspace(X, y, X, 0)
        X_sort = ClassifLibrary.sort_attributes(X)
        treshold = X_sort[0][0] + (X_sort[-1][0] - X_sort[0][0]) / self.NUMBER_OF_SUBSPACES
        # then
        for i in range(len(X_sub)):
            self.assertTrue(X_sub[i] <= treshold)

    def test_should_return_right_minima_and_maxima(self):
        # given
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        # when
        X_0_min, X_0_max, X_1_min, X_1_max = ClassifLibrary.get_samples_limits(X)
        # then
        self.assertEqual(X[0][0], X_0_min)
        self.assertEqual(X[-1][0], X_0_max)
        self.assertEqual(X[0][1], X_1_min)
        self.assertEqual(X[-1][1], X_1_max)

    def test_should_return_right_minima_and_maxima_for_subspace(self):
        # given
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        # when
        X_min, X_max = ClassifLibrary.get_subdata_limits(X)
        # then
        self.assertEqual(X[0][0], X_min)
        self.assertEqual(X[-1][0], X_max)

    def test_should_return_right_number_of_subplots_when_external_plots_drawn(self):
        # given
        data = ClassifierData(show_color_plot = True)
        # when
        target = ClassifLibrary.determine_number_of_subplots(data)
        # then
        self.assertEqual(self.NUMBER_OF_CLASSIFIERS * 2 + 1, target)

    def test_should_return_right_number_of_subplots_when_external_plots_not_drawn(self):
        # given
        # when
        target = ClassifLibrary.determine_number_of_subplots()
        # then
        self.assertEqual(self.NUMBER_OF_CLASSIFIERS + 1, target)

    def test_should_evaluate_average_coefficients_from_n_best(self):
        # given
        coefficients = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        scores = [[0.25], [0], [0.5], [0.75], [1]]
        # when
        a, b = \
            ClassifLibrary.evaluate_average_coefficients_from_n_best(
                coefficients, scores, 0,
                ClassifierData(number_of_best_classifiers = 3, number_of_classifiers = len(scores)))
        # then
        self.assertEqual((coefficients[2][0] + coefficients[3][0] + coefficients[4][0]) / 3, a)
        self.assertEqual((coefficients[2][1] + coefficients[3][1] + coefficients[4][1]) / 3, b)

    def test_should_evaluate_weighted_average_coefficients_from_n_best(self):
        # given
        coefficients = [[1, 2], [3, 4], [5, 6], [7, 8]]
        scores = [[0], [0.25], [0.5], [0.75]]
        # when
        a, b = \
            ClassifLibrary.evaluate_weighted_average_coefficients_from_n_best(
                coefficients, scores, 0,
                ClassifierData(number_of_best_classifiers = 2, number_of_classifiers = len(scores)))
        # then
        self.assertEqual((coefficients[2][0] * scores[2][0] + coefficients[3][0] * scores[3][0]) /
                         (scores[2][0] + scores[3][0]), a)
        self.assertEqual((coefficients[2][1] * scores[2][0] + coefficients[3][1] * scores[3][0]) /
                         (scores[2][0] + scores[3][0]), b)

    def test_should_return_subspace_limits(self):
        # given
        X = np.array([[0.1, 0], [1.3, 1], [2.5, 2], [3.7, 3], [7.9, 7], [8.7, 8], [9.5, 9], [10.3, 10]])
        number_of_subspace = 2
        # when
        x_subspace_max, x_subspace_min = ClassifLibrary.get_subspace_limits(X, number_of_subspace)
        x_expexted_max, x_expexted_min = \
            X[0][0] + (number_of_subspace + 1) * (X[-1][0] - X[0][0]) / self.NUMBER_OF_SUBSPACES, \
            X[0][0] + number_of_subspace * (X[-1][0] - X[0][0]) / self.NUMBER_OF_SUBSPACES
        # then
        self.assertEqual(x_expexted_min, x_subspace_min)
        self.assertEqual(x_expexted_max, x_subspace_max)

    def test_should_return_right_mccs(self):
        # given
        tp, tn, fp, fn = self.conf_matrix[0][0], self.conf_matrix[1][1], self.conf_matrix[1][0], self.conf_matrix[0][1]
        # when
        mcc = ClassifLibrary.compute_mcc([self.conf_matrix])
        # then
        self.assertEqual(len([self.conf_matrix]), len(mcc))
        self.assertEqual((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), mcc[0])

    def test_should_return_no_error_on_default_merging_algorithm(self):
        # given
        # when
        result = apply(ClassifierData())
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat')
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv')
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv')
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv')
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_show_plot(self):
        # given
        # when
        result = apply(ClassifierData(show_plots = True))
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat', show_plots = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv', show_plots = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv', show_plots = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv', show_plots = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin(self):
        # given
        # when
        result = apply(ClassifierData(show_plots = True, show_color_plot = True))
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat', show_plots = True, show_color_plot = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv', show_plots = True, show_color_plot = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv', show_plots = True, show_color_plot = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv', show_plots = True, show_color_plot = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin_computed(self):
        # given
        # when
        result = apply(ClassifierData(show_plots = True, show_color_plot = True, write_computed_scores = True))
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat', show_plots = True, show_color_plot = True, write_computed_scores = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv', show_plots = True, show_color_plot = True, write_computed_scores = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv', show_plots = True, show_color_plot = True, write_computed_scores = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv', show_plots = True, show_color_plot = True, write_computed_scores = True)
        # when
        result = apply(classifier_data)
        # then
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
