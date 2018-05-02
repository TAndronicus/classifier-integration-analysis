import itertools
import math
import warnings
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import xlrd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC

from ClassifierData import ClassifierData
from ClfType import ClfType
from IntegrRes import IntegrRes
from NotEnoughSamplesError import NotEnoughSamplesError


def determine_clf_type(clf):
    """Determines type of classifier

    :param clf:
    :return: clfType: ClfType
    """
    try:
        clf.coef_
        return ClfType.LINEAR
    except:
        # print('No coef_ attribute')
        pass
    try:
        clf.centroids_
        return ClfType.MEAN
    except:
        # print('No centroids_ attribute')
        print('No attribute found')
    raise Exception('Classifier not defined')


def initialize_classifiers(classifier_data: ClassifierData = ClassifierData()):
    """Generates list of classifiers for analysis

    :param classifier_data: ClassifierData
    :return: List of classifiers: [clf, ..., clf]
    """
    type_of_classifier = classifier_data.type_of_classifier
    number_of_classifiers = classifier_data.number_of_classifiers
    clfs = []
    if type_of_classifier == ClfType.LINEAR:
        for i in range(number_of_classifiers):
            clfs.append(LinearSVC(max_iter = 1e6, tol = 1e-10, C = 100))
    elif type_of_classifier == ClfType.MEAN:
        for i in range(number_of_classifiers):
            clfs.append(NearestCentroid())
    else:
        raise Exception('Classifier type not defined')
    return clfs


def prepare_raw_data(classifier_data: ClassifierData = ClassifierData()):
    """Prepares raw data for classification

    :param classifier_data: ClassifierData
    :return: X, y: np.array, np.array
    """
    are_samples_generated = classifier_data.are_samples_generated
    is_validation_hard = classifier_data.is_validation_hard
    if are_samples_generated:
        number_of_samples_if_generated = classifier_data.number_of_samples_if_generated
        X, y = make_classification(n_features = 2, n_redundant = 0, n_informative = 1, n_clusters_per_class = 1,
                                   n_samples = number_of_samples_if_generated,
                                   class_sep = 2.7, hypercube = False, random_state = 2)
        X0, X1 = divide_generated_samples(X, y)
        X0, X1 = sort_attributes(X0), sort_attributes(X1)
        if is_validation_hard:
            assert_distribution(X0, X1, classifier_data)
        else:
            assert_distribution_simplified(X0, X1, classifier_data)
        return compose_sorted_parts(X0, X1)
    else:
        return load_samples_from_datasets(classifier_data)


def load_two_first_columns_preincrement(sheet: xlrd.sheet.Sheet):
    """Reads dataset with X from two first columns, skips first row

    :param sheet: xlrd.sheet.Sheet
    :return: X, y: [], []
    """
    line_number = 0
    line = sheet.row(line_number)
    number_of_columns = len(line)
    X, y = np.zeros((sheet.nrows, 2)), np.zeros(sheet.nrows, dtype = np.int)
    while line_number < sheet.nrows - 1:
        line_number += 1
        line = sheet.row(line_number)
        row = []
        for i in range(2):  # range(number_of_columns - 1):
            row.append(float(line[i].value))
        X[line_number - 1, :] = row
        y[line_number - 1] = int(line[number_of_columns - 1].value)
    return X, y


def read_features(sheet: xlrd.sheet.Sheet, classifier_data: ClassifierData = ClassifierData()):
    """Reads rows of data from file

    :param sheet: xlrd.sheet.Sheet
    :param classifier_data: ClassifierData
    :return: X0, X1: [], []
    """
    columns = classifier_data.columns
    line_number = 0
    line = sheet.row(line_number)
    number_of_columns = len(line)
    X0, X1 = [], []
    while line_number < sheet.nrows:
        line = sheet.row(line_number)
        row = []
        for i in columns:
            row.append(float(line[i].value))
        if int(line[number_of_columns - 1].value) == 0:
            X0.append(row)
        else:
            X1.append(row)
        line_number += 1
    return X0, X1


def read_excel_file(classifier_data: ClassifierData = ClassifierData()):
    """Read rows of data and make a selection

    :param classifier_data: ClassifierData
    :return: X, y: [], []
    """
    filename = classifier_data.filename
    number_of_dataset_if_not_generated = classifier_data.number_of_dataset_if_not_generated
    file = xlrd.open_workbook(filename)
    sheet = file.sheet_by_index(number_of_dataset_if_not_generated)
    line_number = 0
    line = sheet.row(line_number)
    number_of_columns = len(line)
    X, y = np.zeros((sheet.nrows, number_of_columns - 1)), np.zeros(sheet.nrows, dtype = np.int)
    while line_number < sheet.nrows:
        line = sheet.row(line_number)
        row = []
        for i in range(number_of_columns - 1):
            row.append(float(line[i].value))
        X[line_number - 1, :] = row
        y[line_number - 1] = int(float(line[number_of_columns - 1].value))
        line_number += 1
    return X, y


def get_columns_from_scores(feature_scores: []):
    """Extracts vector with columns of best features

    :param feature_scores: []
    :return: columns: []
    """
    first_maximum, second_maximum, first_column, second_column = 0, 0, 0, 0
    for i in range(len(feature_scores)):
        if feature_scores[i] > first_maximum:
            second_column = first_column
            first_column = i
            second_maximum = first_maximum
            first_maximum = feature_scores[i]
        elif feature_scores[i] > second_maximum:
            second_column = i
            second_maximum = feature_scores[i]
    print('Columns selected: {} and {}'.format(first_column, second_column))
    return [first_column, second_column]


def get_separate_columns(X: [], y: [], columns: []):
    """Extract given columns from data

    :param X: []
    :param y: []
    :param columns: []
    :return: X0, X1: [], []
    """
    X0, X1 = [], []
    for i in range(len(X)):
        row = []
        for column in columns:
            row.append(X[i][column])
        if y[i] == 0:
            X0.append(row)
        else:
            X1.append(row)
    return X0, X1


def load_samples_from_file_non_parametrized(filename: str):
    """Loads data from file, takes two first columns, skips first row

    Warning: method does not sort data
    Missing sorting and composing
    :param filename: Name of file
    :return: X, y: np.array, np.array - samples for classification
    """
    file = xlrd.open_workbook(filename)
    sheet = file.sheet_by_index(0)
    X, y = load_two_first_columns_preincrement(sheet)
    return X, y


def load_samples_from_datasets_first_two_rows(classifier_data: ClassifierData = ClassifierData()):
    """Loads data from dataset (xlsx file with data)

    :param classifier_data: ClassifierData
    :return: X, y: np.array, np.array - samples for classification
    """
    number_of_dataset_if_not_generated = classifier_data.number_of_dataset_if_not_generated
    file = xlrd.open_workbook('datasets.xlsx')
    sheet = file.sheet_by_index(number_of_dataset_if_not_generated)
    columns = classifier_data.columns
    classifier_data.columns = [0, 1]
    X0, X1 = read_features(sheet, classifier_data)
    classifier_data.columns = columns
    # X = SelectKBest(k = 2).fit_transform(X, y)
    print('Ratio (0:1): {}:{}'.format(len(X0), len(X1)))
    X0, X1 = sort_attributes(X0), sort_attributes(X1)
    assert_distribution_simplified(X0, X1, classifier_data)
    X, y = compose_sorted_parts(X0, X1)
    return X, y


def load_samples_from_datasets_non_parametrised(classifier_data: ClassifierData = ClassifierData()):
    """Loads data from dataset (xlsx file with data)

    :param classifier_data: ClassifierData
    :return: X, y: np.array, np.array - samples for classification
    """
    number_of_dataset_if_not_generated = classifier_data.number_of_dataset_if_not_generated
    file = xlrd.open_workbook('datasets.xlsx')
    sheet = file.sheet_by_index(number_of_dataset_if_not_generated)
    X0, X1 = read_features(sheet, classifier_data)
    print('Ratio (0:1): {}:{}'.format(len(X0), len(X1)))
    X0, X1 = sort_attributes(X0), sort_attributes(X1)
    assert_distribution_simplified(X0, X1, classifier_data)
    X, y = compose_sorted_parts(X0, X1)
    return X, y


def load_samples_from_datasets(classifier_data: ClassifierData = ClassifierData()):
    """Loads data from dataset (xlsx file with data)

    :param classifier_data: ClassifierData
    :return: X, y: np.array, np.array - samples for classification
    """
    is_validation_hard = classifier_data.is_validation_hard
    filename = classifier_data.filename
    if filename.endswith(".dat") or filename.endswith(".csv"):
        X, y = read_csv_file(classifier_data)
    elif filename.endswith(".tsv"):
        X, y = read_sv_file(classifier_data, '\t')
    elif filename.endswith(".scsv"):
        X, y = read_sv_file(classifier_data, ';')
    else:
        X, y = read_excel_file(classifier_data)
    X0, X1 = make_selection(X, y, classifier_data)
    print('Ratio (0:1): {}:{}'.format(len(X0), len(X1)))
    X0, X1 = sort_attributes(X0), sort_attributes(X1)
    if is_validation_hard:
        assert_distribution(X0, X1, classifier_data)
    else:
        assert_distribution_simplified(X0, X1, classifier_data)
    X, y = compose_sorted_parts(X0, X1)
    return X, y


def read_csv_file(classifier_data: ClassifierData = ClassifierData()):
    """Reads data from comma separated value files

    :param classifier_data: ClassifierData
    :return: X, y: [], []
    """
    filename = classifier_data.filename
    file = open(filename, "r")
    lines = file.readlines()
    file.close()
    X, y = [], []
    for line in lines:
        if line.startswith("@"):
            continue
        line_as_array = line.replace("\n", "", 1).split(",")
        row = []
        for i in range(len(line_as_array) - 1):
            row.append(float(line_as_array[i]))
        X.append(row)
        y.append(int(float(line_as_array[-1])))
    return X, y


def read_sv_file(classifier_data: ClassifierData = ClassifierData(), separator: str = ','):
    """Reads data from tab separated value files

    :param classifier_data: ClassifierData
    :param separator: str
    :return: X, y: [], []
    """
    filename = classifier_data.filename
    file = open(filename, "r")
    lines = file.readlines()
    file.close()
    X, y = [], []
    for line in lines:
        line_as_array = line.replace("\n", "", 1).split(separator)
        row = []
        for i in range(len(line_as_array) - 1):
            row.append(float(line_as_array[i]))
        X.append(row)
        y.append(int(float(line_as_array[-1])))
    return X, y


def make_selection(X: [], y: [], classifier_data: ClassifierData = ClassifierData()):
    """Returns 2 best columns in 2 datasets for each class

    :param X: []
    :param y: []
    :param classifier_data: ClassifierData
    :return: X0, X1: [], []
    """
    switch_columns_while_loading = classifier_data.switch_columns_while_loading
    selection = SelectKBest(k = 2, score_func = f_classif)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selection.fit(X, y)
    feature_scores = selection.scores_
    columns = get_columns_from_scores(feature_scores)
    if switch_columns_while_loading:
        columns[0], columns[1] = columns[1], columns[0]
        print('Columns switched: {} and {}'.format(columns[0], columns[1]))
    return get_separate_columns(X, y, columns)


def assert_distribution(X0: [], X1: [], classifier_data: ClassifierData = ClassifierData()):
    """Asserts that samples can be divided into subspaces with the same amount of data

    :param X0: np.array, data with class 0
    :param X1: np.array, data with class 1
    :param classifier_data: ClassifierData
    data prepared to be divided into number_of_classifiers + 2 parts of same length
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    number_of_classifiers = classifier_data.number_of_classifiers
    print('len0: {}, len1: {}'.format(len(X0), len(X1)))
    x0_min, x0_max = get_subdata_limits(X0)
    x1_min, x1_max = get_subdata_limits(X1)
    x_min, x_max = min(x0_min, x1_min), max(x0_max, x1_max)
    for i in range(number_of_space_parts):
        counter0, counter1, index0, index1 = 0, 0, 0, 0
        while True:
            if index0 == len(X0):
                break
            if x_min + i * (x_max - x_min) / number_of_space_parts <= X0[index0][0] <= \
                    x_min + (i + 1) * (x_max - x_min) / number_of_space_parts:
                counter0 += 1
                index0 += 1
                continue
            if counter0 > 0:
                break
            index0 += 1
        while True:
            if index1 == len(X1):
                break
            if x_min + i * (x_max - x_min) / number_of_space_parts <= X1[index1][0] <= \
                    x_min + (i + 1) * (x_max - x_min) / number_of_space_parts:
                counter1 += 1
                index1 += 1
                continue
            if counter1 > 0:
                break
            index1 += 1
        if counter0 + counter1 < number_of_classifiers + 2:
            print('Only {} samples in {}. subspace'.format(counter0 + counter1, i + 1))
            from_front = False
            return cut_out_from_larger(X0, X1, from_front, classifier_data)
        remainder = (counter0 + counter1) % (number_of_classifiers + 2)
        if remainder != 0:
            if i == 0:
                from_front = True
                return cut_out_from_larger(X0, X1, from_front, classifier_data)
            if i == number_of_space_parts - 1:
                from_front = False
                return cut_out_from_larger(X0, X1, from_front, classifier_data)
            if len(X0) > len(X1):
                if counter0 < remainder:
                    subtraction, rest = counter0, remainder - counter0
                else:
                    subtraction, rest = remainder, 0
                X0 = np.vstack((X0[:index0 - subtraction], X0[index0:]))
                X1 = np.vstack((X1[:index1 - rest], X1[index1:]))
            else:
                if counter1 < remainder:
                    subtraction, rest = counter1, remainder - counter1
                else:
                    subtraction, rest = remainder, 0
                X0 = np.vstack((X0[:index0 - rest], X0[index0:]))
                X1 = np.vstack((X1[:index1 - subtraction], X1[index1:]))
    print('len0: {}, len1: {}'.format(len(X0), len(X1)))
    return X0, X1


def cut_out_from_larger(X0: [], X1: [], from_front: bool, classifier_data: ClassifierData = ClassifierData()):
    """Cuts value from set

    :param X0: []
    :param X1: []
    :param from_front: bool
    :param classifier_data: ClassifierData
    :return:
    """
    if from_front:
        if len(X0) > len(X1):
            return assert_distribution(X0[1:], X1, classifier_data)
        return assert_distribution(X0, X1[1:], classifier_data)
    else:
        if len(X0) > len(X1):
            return assert_distribution(X0[:-1], X1, classifier_data)
        return assert_distribution(X0, X1[:-1], classifier_data)


def assert_distribution_simplified(X0: [], X1: [], classifier_data: ClassifierData = ClassifierData()):
    """Asserts that samples can be divided into subspaces with the same amount of data

    :param X0: np.array, data with class 0
    :param X1: np.array, data with class 1
    :param classifier_data: ClassifierData
    :return: X0, X1: np.array, np.array, data prepared to be divided into number_of_classifiers + 2 parts of same length
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    number_of_classifiers = classifier_data.number_of_classifiers
    print('Validating dataset')
    x_min, x_max = get_extrema_for_subspaces(X0, X1)
    previous_index0, previous_index1 = 0, 0
    print('Before assertion: len0: {}, len1: {}'.format(len(X0), len(X1)))
    for i in range(number_of_space_parts):
        counter0, index0 = get_count_of_samples_in_subspace_and_beg_ind_of_next_subspace(X0, x_min, x_max, i,
                                                                                         classifier_data)
        counter1, index1 = get_count_of_samples_in_subspace_and_beg_ind_of_next_subspace(X1, x_min, x_max, i,
                                                                                         classifier_data)
        if counter0 + counter1 < number_of_classifiers + 2:
            print('Only {} samples in {}. subspace'.format(counter0 + counter1, i + 1))
        remainder = (counter0 + counter1) % (number_of_classifiers + 2)
        if remainder != 0:
            if i != number_of_space_parts - 1:
                relative_index0, relative_index1, is_last = index0, index1, False
            else:
                relative_index0, relative_index1, is_last = previous_index0, previous_index1, True
            if counter0 > counter1:
                counter, is_first_bigger = counter0, True
            else:
                counter, is_first_bigger = counter1, False
            X0, X1 = limit_datasets(X0, X1, counter, remainder, relative_index0, relative_index1, is_first_bigger,
                                    is_last)
        previous_index0, previous_index1 = index0, index1
    print('After assertion: len0: {}, len1: {}'.format(len(X0), len(X1)))
    return X0, X1


def get_extrema_for_subspaces(X0: [], X1: []):
    """Returns minimum and maximum bor both datasets

    :param X0: np.array
    :param X1: np.array
    :return: minimum, maximum: float, float
    """
    x0_min, x0_max = get_subdata_limits(X0)
    x1_min, x1_max = get_subdata_limits(X1)
    return min(x0_min, x1_min), max(x0_max, x1_max)


def get_count_of_samples_in_subspace_and_beg_ind_of_next_subspace(X0: [], x_min: float, x_max: float, i: int,
                                                                  classifier_data: ClassifierData = ClassifierData()):
    """Returns of count of data in subspace as well as index of the first element of next subspace

    :param X0: np.array
    :param x_min: float
    :param x_max: float
    :param i: int, number of subspace - 1
    :param classifier_data: ClassifierData
    :return: counter, index: int, int
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    counter, index = 0, 0
    while True:
        if index == len(X0):
            break
        if x_min + i * (x_max - x_min) / number_of_space_parts <= X0[index][0] <= x_min + (i + 1) * (x_max - x_min) / \
                number_of_space_parts:
            counter += 1
            index += 1
            continue
        if counter > 0:
            break
        index += 1
    return counter, index


def set_subtraction_and_rest(counter: int, remainder: int):
    """Calculates substracion and rest for limiting datasets

    :param counter: int
    :param remainder: int
    :return: substraction, rest: int, int
    """
    if counter < remainder:
        return counter, remainder - counter
    else:
        return remainder, 0


def limit_datasets_for_every_subspace_but_last(X0: [], X1: [], counter: int, remainder: int, index0: int,
                                               index1: int, is_first_bigger: bool):
    """Returns limited datasets for every subspace but last one

    :param X0: np.array
    :param X1: np.array
    :param counter: int
    :param remainder: int
    :param index0: int
    :param index1: int
    :param is_first_bigger: boolean, true if there is more data in analysiert subspace in the first dataset
    :return: X0, X1: np.array, np.array
    """
    subtraction, rest = set_subtraction_and_rest(counter, remainder)
    if is_first_bigger:
        first_subtraction, second_subtraction = subtraction, rest
    else:
        first_subtraction, second_subtraction = rest, subtraction
    X0 = np.vstack((X0[:index0 - first_subtraction], X0[index0:]))
    X1 = np.vstack((X1[:index1 - second_subtraction], X1[index1:]))
    return X0, X1


def limit_datasets_for_last_subspace(X0: [], X1: [], counter: int, remainder: int, previous_index0: int,
                                     previous_index1: int, is_first_bigger: bool):
    """Returns limited datasets for last subspace

    :param X0: np.array
    :param X1: np.array
    :param counter: int
    :param remainder: int
    :param previous_index0: int
    :param previous_index1: int
    :param is_first_bigger: boolean, true if there is more data in analysiert subspace in the first dataset
    :return: X0, X1: np.array, np.array
    """
    subtraction, rest = set_subtraction_and_rest(counter, remainder)
    if is_first_bigger:
        first_subtraction, second_subtraction = subtraction, rest
    else:
        first_subtraction, second_subtraction = rest, subtraction
    X0 = np.vstack((X0[:previous_index0], X0[previous_index0 + first_subtraction:]))
    X1 = np.vstack((X1[:previous_index1], X1[previous_index1 + second_subtraction:]))
    return X0, X1


def limit_datasets(X0: [], X1: [], counter: int, remainder: int, relative_index0: int, relative_index1: int,
                   is_first_bigger: bool, is_last: bool):
    """Returns limited datasets

    :param X0: np.array
    :param X1: np.array
    :param counter: int
    :param remainder: int
    :param relative_index0: int
    :param relative_index1: int
    :param is_first_bigger: boolean
    :param is_last: boolean, true if it is last subspace
    :return: X0, X1: np.array, np.array
    """
    if is_last:
        return limit_datasets_for_last_subspace(X0, X1, counter, remainder, relative_index0, relative_index1,
                                                is_first_bigger)
    else:
        return limit_datasets_for_every_subspace_but_last(X0, X1, counter, remainder, relative_index0, relative_index1,
                                                          is_first_bigger)


def compose_sorted_parts(X0: [], X1: []):
    """Composes classification data from 1 and 0 parts, datasets must be sorted

    :param X0: [], data, where y = 0
    :param X1: [], data, where y = 1
    :return: X, y: np.array, np.array - samples for classification
    """
    X, y = np.zeros((len(X0) + len(X1), 2)), np.zeros(len(X0) + len(X1), dtype = np.int)
    for i in range(len(X0)):
        X[i, :], y[i] = X0[i], 0
    for i in range(len(X1)):
        X[len(X0) + i, :], y[len(X0) + i] = X1[i], 1
    return X, y


def sort_attributes(X: []):
    """Sorts attribute array

    :param X: []
    :return: X: np.array
    """
    length, X_result_array = len(X), []
    for input_index in range(length):
        output_index = 0
        while output_index < input_index:
            if X_result_array[output_index][0] > X[input_index][0]:
                break
            output_index += 1
        X_result_array.insert(output_index, X[input_index])
    X_result = np.zeros((length, 2))
    for i in range(length):
        X_result[i, :] = X_result_array[i]
    return X_result


def sort_results(X: [], y: []):
    """Sorts attribute and class arrays

    Warning: does not sort by classes (y is not taken into consideration)
    :param X: np.array
    :param y: np.array
    :return: X, y: np.array, np.array
    """
    length = len(y)
    X_result_array, y_result_array = [], []
    for input_index in range(length):
        output_index = 0
        while output_index < input_index:
            if X_result_array[output_index][0] > X[input_index, 0]:
                break
            output_index += 1
        X_result_array.insert(output_index, X[input_index, :])
        y_result_array.insert(output_index, y[input_index])
    X_result, y_result = np.zeros((length, 2)), np.zeros(length, dtype = np.int)
    for i in range(length):
        X_result[i, :] = X_result_array[i]
        y_result[i] = y_result_array[i]
    return X_result, y_result


def divide_generated_samples(X: [], y: []):
    """Divide samples into arrays X0 (with y = 0) and X1 (y = 1)

    :param X: np.array
    :param y: np.array
    :return: X0, X1: [], []
    """
    X0, X1 = [], []
    for i in range(len(y)):
        if y[i] == 0:
            X0.append(X[i, :])
        else:
            X1.append(X[i, :])
    return X0, X1


def divide_samples_between_classifiers(X: [], y: [], classifier_data: ClassifierData = ClassifierData()):
    """Divides sample into parts for every classifier

    Warning: does not take sorting into consideration
    Prepares n = number_of_classifiers + 1 parts X(k) and y(k) with the same count
    and returns [X(1), X(2), ..., X(n - 1)], [y(1), y(2), ..., y(n - 1)], X(n), y(n)
    :param X: np.array, raw data
    :param y: np.array, raw data
    :param classifier_data: ClassifierData
    :return: [X(1), X(2), ..., X(n - 1)], [y(1), y(2), ..., y(n - 1)], X(n), y(n)
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    number_of_samples = len(X)
    X_whole, y_whole, X_rest, y_rest = [], [], [], []
    X_final_test, X_rest, y_final_test, y_rest = \
        train_test_split(X, y, train_size = int(number_of_samples / (number_of_classifiers + 1)))
    for i in range(number_of_classifiers - 1):
        X_part, X_rest, y_part, y_rest = \
            train_test_split(X_rest, y_rest, train_size = int(number_of_samples / (number_of_classifiers + 1)))
        X_whole.append(X_part)
        y_whole.append(y_part)
    X_whole.append(X_rest)
    y_whole.append(y_rest)
    return X_whole, y_whole, X_final_test, y_final_test


def split_sorted_samples_between_classifiers(X: [], y: [], classifier_data: ClassifierData = ClassifierData()):
    """Splits sorted samples between classifiers

    :param X: np.array
    :param y: np.array
    :param classifier_data: ClassifierData
    :return: X_whole, y_whole, X_final_test, y_final_test:
    [np.array(1), np.array(2), ..., np.array(number_of_classifiers)],
    [np.array(1), np.array(2), ..., np.array(number_of_classifiers)], np.array, np.array
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    length = int(len(X) / (number_of_classifiers + 1))
    X_whole, y_whole, X_final_test, y_final_test = [], [], np.zeros((length, 2)), np.zeros(length, dtype = np.int)
    for i in range(number_of_classifiers):
        X_temp, y_temp = np.zeros((length, 2)), np.zeros(length, dtype = np.int)
        for j in range(length):
            X_temp[j, :] = (X[j * (number_of_classifiers + 1) + i, :])
            y_temp[j] = (y[j * (number_of_classifiers + 1) + i])
        X_whole.append(X_temp)
        y_whole.append(y_temp)
    for i in range(length):
        X_final_test[i, :] = (X[(i + 1) * number_of_classifiers + i, :])
        y_final_test[i] = (y[(i + 1) * number_of_classifiers + i])
    return X_whole, y_whole, X_final_test, y_final_test


def divide_samples_between_training_and_testing(X_unsplitted: [], y_unsplitted: [], quotient: float = 2 / 3):
    """Divides sample into parts for ttaining and testing

    Warning: does not take sorting into consideration
    :param X_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param y_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param quotient: float
    :return: X_train, X_test, y_train, y_test:
    [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)],
    [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)]
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    for X_one, y_one in zip(X_unsplitted, y_unsplitted):
        X_train_part, X_test_part, y_train_part, y_test_part = \
            train_test_split(X_one, y_one, train_size = int(len(X_one) * quotient))
        X_train.append(X_train_part)
        X_test.append(X_test_part)
        y_train.append(y_train_part)
        y_test.append(y_test_part)
    return X_train, X_test, y_train, y_test


def split_sorted_samples_between_training_and_testing(X_unsplitted: [], y_unsplitted: [], quotient: float = 2 / 3):
    """Splits sorted samples for testing and training

    :param X_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param y_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param quotient: float
    :return: X_train, X_test, y_train, y_test:
    [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)],
    [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)]
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    for X_one, y_one in zip(X_unsplitted, y_unsplitted):
        X_train_part, X_test_part, y_train_part, y_test_part = train_test_sorted_split(X_one, y_one, quotient)
        X_train.append(X_train_part)
        X_test.append(X_test_part)
        y_train.append(y_train_part)
        y_test.append(y_test_part)
    return X_train, X_test, y_train, y_test


def split_sorted_samples(X: [], y: [], classifier_data: ClassifierData = ClassifierData()):
    """Splits raw data into number_of_classifiers + 2 parts of same length

    :param X: np.array, array with sorted data
    :param y: np.array
    :param classifier_data: ClassifierData
    :return: X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test:
    [np.array(1), np.array(2), ..., np.array(number_of_classifiers)],
    [np.array(1), np.array(2), ..., np.array(number_of_classifiers)], np.array, np.array, np.array, np.array
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    number_of_space_parts = classifier_data.number_of_space_parts
    print('Splitting samples')
    if len(X) < (number_of_classifiers + 2) * number_of_space_parts:
        raise NotEnoughSamplesError('Not enough samples found when sorting (len(X) = {})'.format(len(X)))
    length = int(len(X) / (number_of_classifiers + 2))
    X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
        [], [], np.zeros((length, 2)), np.zeros(length, dtype = np.int), np.zeros((length, 2)), \
        np.zeros(length, dtype = np.int)
    for i in range(number_of_classifiers):
        X_temp, y_temp = np.zeros((length, 2)), np.zeros(length, dtype = np.int)
        for j in range(length):
            X_temp[j, :] = (X[j * (number_of_classifiers + 2) + i, :])
            y_temp[j] = (y[j * (number_of_classifiers + 2) + i])
        X_whole_train.append(X_temp)
        y_whole_train.append(y_temp)
    for i in range(length):
        X_validation[i, :] = (X[(i + 1) * (number_of_classifiers + 2) - 2, :])
        y_validation[i] = (y[(i + 1) * (number_of_classifiers + 2) - 2])
    for i in range(length):
        X_test[i, :] = (X[(i + 1) * (number_of_classifiers + 2) - 1, :])
        y_test[i] = (y[(i + 1) * (number_of_classifiers + 2) - 1])
    return X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test


def split_sorted_unitary_bagging(X: [], y: [], classifier_data: ClassifierData = ClassifierData()):
    """Splits data into subsets for training, validating and testing using bagging

    :param X: []
    :param y: []
    :param classifier_data: ClassifierData
    :return: X_splitted, y_splitted: [], []
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    number_of_space_parts = classifier_data.number_of_space_parts
    print('Splitting samples')
    if len(X) < (number_of_classifiers + 2) * number_of_space_parts:
        raise NotEnoughSamplesError('Not enough samples found when sorting (len(X) = {})'.format(len(X)))
    X_splitted, y_splitted = [], []
    length = len(X)
    for i in range(number_of_classifiers + 2):
        X_temp, y_temp = np.zeros((length, 2)), np.zeros(length, dtype = np.int)
        for j in range(length):
            rand = randint(0, length - 1)
            X_temp[j, :] = X[rand, :]
            try:
                if X[rand, 0] < minimum:
                    minimum = X[rand, 0]
            except UnboundLocalError:
                minimum = X[rand, 0]
            try:
                if X[rand, 0] > maximum:
                    maximum = X[rand, 0]
            except UnboundLocalError:
                maximum = X[rand, 0]
            y_temp[j] = y[rand]
        X_splitted.append(X_temp)
        y_splitted.append(y_temp)
    classifier_data.minimum = minimum
    classifier_data.maximum = maximum
    return X_splitted, y_splitted


def split_sorted_unitary(X: [], y: [], classifier_data: ClassifierData = ClassifierData()):
    """Splits data into subsets for training, validating and testing

    :param X: []
    :param y: []
    :param classifier_data: ClassifierData
    :return: X_splitted, y_splitted: [], []
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    number_of_space_parts = classifier_data.number_of_space_parts
    print('Splitting samples')
    if len(X) < (number_of_classifiers + 2) * number_of_space_parts:
        raise NotEnoughSamplesError('Not enough samples found when sorting (len(X) = {})'.format(len(X)))
    X_splitted, y_splitted = [], []
    length_of_subset = int(len(X) / (number_of_classifiers + 2))
    for i in range(number_of_classifiers + 2):
        X_temp, y_temp = np.zeros((length_of_subset, 2)), np.zeros(length_of_subset, dtype = np.int)
        for j in range(length_of_subset):
            X_temp[j, :] = (X[j * (number_of_classifiers + 2) + i, :])
            y_temp[j] = (y[j * (number_of_classifiers + 2) + i])
        X_splitted.append(X_temp)
        y_splitted.append(y_temp)
    classifier_data.minimum, classifier_data.maximum = min(X[:, 0]), max(X[:, 0])
    return X_splitted, y_splitted


def train_test_sorted_split(X_one: [], y_one: [], quotient: float = 2 / 3):
    """Splits dataset into training and testing

    :param X_one: np.array
    :param y_one: np.array
    :param quotient: float
    :return: X_train, X_test, y_train, y_test: np.array, np.array, np.array, np.array
    """
    quotient_freq = 1 / (1 - quotient)
    if quotient_freq - int(quotient_freq) - 1 < 1e-3:
        quotient_freq = int(quotient_freq) + 1
    else:
        quotient_freq = int(quotient_freq)
    length = int(len(y_one) / quotient_freq)
    X_train, X_test, y_train, y_test = np.zeros((length * (quotient_freq - 1), 2)), np.zeros((length, 2)), \
                                       np.zeros(length * (quotient_freq - 1), dtype = np.int), \
                                       np.zeros(length, dtype = np.int)
    counter = 0
    for i in range(length):
        for j in range(quotient_freq - 1):
            X_train[counter, :] = X_one[i * quotient_freq + j, :]
            y_train[counter] = y_one[i * quotient_freq + j]
            counter += 1
        X_test[i, :] = X_one[(i + 1) * quotient_freq - 1]
        y_test[i] = y_one[(i + 1) * quotient_freq - 1]
    return X_train, X_test, y_train, y_test


def prepare_samples_for_subspace(X_test: [], y_test: [], j: int,
                                 classifier_data: ClassifierData = ClassifierData()):
    """Preparing sample for testing in j-th subspace

    :param X_test: np.array
    :param y_test: np.array
    :param j: int
    :param classifier_data: ClassifierData
    :return: X_part, y_part: [], []
    """
    x_samp_max, x_samp_min = get_subspace_limits(j, classifier_data)
    X_part = [row for row in X_test if x_samp_min <= row[0] < x_samp_max]
    y_part = [y_test[k] for k in range(len(y_test)) if x_samp_min <= X_test[k][0] < x_samp_max]
    return X_part, y_part


def get_samples_limits(X: []):
    """Gets limits of j-th subspace

    :param X: np.array
    :return: X_0_min, X_0_max, X_1_min, X_1_max: float, float, float, float
    """
    return X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()


def get_subdata_limits(X: []):
    """Gets limits of one dimensional dataset

    :param X: np.array
    :return: minimum, maximum: float, float
    """
    minimum, maximum = X[0][0], X[0][0]
    for i in range(len(X)):
        if X[i][0] < minimum:
            minimum = X[i][0]
        if X[i][0] > maximum:
            maximum = X[i][0]
    return minimum, maximum


def get_plot_data(X: []):
    """Prepares data for plot generation

    :param X: np.array
    :return: xx, yy, x_min_plot, x_max_plot: dnarray, dnarray, float, float
    """
    x_min, x_max, y_min, y_max = get_samples_limits(X)
    x_shift = 0.1 * (x_max - x_min)
    y_shift = 0.1 * (y_max - y_min)
    x_min_plot, x_max_plot, y_min_plot, y_max_plot = x_min - x_shift, x_max + x_shift, y_min - y_shift, y_max + y_shift
    plot_mesh_step_size = min(x_max_plot - x_min_plot, y_max_plot - y_min_plot) / 2
    xx, yy = np.meshgrid(np.arange(x_min_plot, x_max_plot, plot_mesh_step_size),
                         np.arange(y_min_plot, y_max_plot, plot_mesh_step_size))
    return xx, yy, x_min_plot, x_max_plot, y_min_plot, y_max_plot


def determine_number_of_subplots(classifier_data: ClassifierData = ClassifierData()):
    """Checks number of subplots to draw
    
    :param classifier_data: ClassifierData
    :return: int, number of subplots to draw
    """
    draw_color_plot = classifier_data.draw_color_plot
    number_of_classifiers = classifier_data.number_of_classifiers
    if draw_color_plot:
        return number_of_classifiers * 2 + 1
    return number_of_classifiers + 1


def train_classifiers(clfs: [], X_whole_train: [], y_whole_train: [], X: [], number_of_subplots: int,
                      classifier_data: ClassifierData = ClassifierData()):
    """Trains classifiers

    :param clfs: [], scikit classifiers
    :param X_whole_train: [np.array(1), np.array(2), ..., np.array(number_of_classifiers)]
    :param y_whole_train: [np.array(1), np.array(2), ..., np.array(number_of_classifiers)]
    :param X: np.array
    :param number_of_subplots: int
    :param classifier_data: ClassifierData
    :return: trained_clfs, coefficients: [], []
    """
    type_of_classifier = classifier_data.type_of_classifier
    draw_color_plot = classifier_data.draw_color_plot
    show_plots = classifier_data.show_plots

    print('Training classifiers')
    trained_clfs, coefficients, current_subplot = [], [], 1
    for clf, X_train, y_train in zip(clfs, X_whole_train, y_whole_train):
        clf.fit(X_train, y_train)
        trained_clfs.append(clf)

        if type_of_classifier == ClfType.LINEAR:
            a, b = extract_coefficients_for_linear(clf)
        elif type_of_classifier == ClfType.MEAN:
            a, b = extract_coefficients_for_mean(clf)

        coefficients.append([a, b])

        # Prepare plot
        if show_plots:
            prepare_train_plot(X, X_train, y_train, a, b, number_of_subplots, current_subplot)
            current_subplot += 1

        # Draw color plot
        if show_plots and draw_color_plot:
            prepare_builtin_train_plot(X, clf, current_subplot, number_of_subplots)
            current_subplot += 1
    return trained_clfs, coefficients


def prepare_train_plot(X: [], X_train: [], y_train: [], a: float, b: float, number_of_subplots: int,
                       current_subplot: int):
    """

    :param X: []
    :param X_train: []
    :param y_train: []
    :param a: float
    :param b: floar
    :param number_of_subplots: int
    :param current_subplot: int
    :return:
    """
    xx, yy, x_min_plot, x_max_plot, y_min_plot, y_max_plot = get_plot_data(X)
    ax = plt.subplot(1, number_of_subplots, current_subplot)
    ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
    x = np.linspace(x_min_plot, x_max_plot)
    y = a * x + b
    ax.plot(x, y)
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(y_min_plot, y_max_plot)


def prepare_builtin_train_plot(X: [], clf: [], current_subplot: int, number_of_subplots: int):
    """Prepares plot built in scikit learn

    :param X: []
    :param clf: []
    :param current_subplot: int
    :param number_of_subplots: int
    :return:
    """
    xx, yy, x_min_plot, x_max_plot, y_min_plot, y_max_plot = get_plot_data(X)
    ax = plt.subplot(1, number_of_subplots, current_subplot)
    if hasattr(clf, 'decision_function'):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    elif hasattr(clf, 'predict_proba'):
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha = .8)


def extract_coefficients_for_linear(clf):
    """Gets a and b coefficients of the linear classifier

    :param clf: Clf
    :return: a, b: float, float
    """
    a, b = - clf.coef_[0][0] / clf.coef_[0][1], - clf.intercept_[0] / clf.coef_[0][1]
    return a, b


def extract_coefficients_for_mean(clf):
    """Extracts coefficients for mean classifier

    :param clf: Clf
    :return: a, b: float, float
    """
    x1, x2, y1, y2 = clf.centroids_[0][0], clf.centroids_[1][0], clf.centroids_[0][1], clf.centroids_[1][1]
    a_t = (y2 - y1) / (x2 - x1)
    a = -1 / a_t
    x3, y3 = (x1 + x2) / 2, (y1 + y2) / 2
    b = y3 - a * x3
    return a, b


def evaluate_average_coefficients_from_n_best(coefficients: [], scores: [], j: int,
                                              classifier_data: ClassifierData = ClassifierData()):
    """Evaluates coefficients from n best classifiers in j-th subspace

    :param coefficients: []
    :param scores: []
    :param j: int
    :param classifier_data: ClassifierData
    :return: a, b: float, float
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    number_of_best_classifiers = classifier_data.number_of_best_classifiers
    a, b, params = 0, 0, []
    for i in range(number_of_classifiers):
        params.append([scores[i][j], coefficients[i]])
    params.sort()
    for i in range(number_of_best_classifiers):
        a += params[number_of_classifiers - 1 - i][1][0]
        b += params[number_of_classifiers - 1 - i][1][1]
    return a / number_of_best_classifiers, b / number_of_best_classifiers


def evaluate_weighted_average_coefficients_from_n_best(coefficients: [], scores: [], j: int,
                                                       classifier_data: ClassifierData = ClassifierData()):
    """Evaluates coefficients from n best classifiers in j-th subspace

    :param coefficients: []
    :param scores: []
    :param j: int
    :param classifier_data: ClassifierData
    :return: a, b: float, float
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    number_of_best_classifiers = classifier_data.number_of_best_classifiers
    a, b, scoreSum, scores_in_subspace = 0, 0, 0, np.zeros(number_of_classifiers)
    for i in range(number_of_classifiers):
        scores_in_subspace[i] = scores[i][j]
    indices = list(range(number_of_classifiers))
    sorted_scores, indices = (list(t) for t in zip(*sorted(zip(scores_in_subspace, indices))))
    for i in range(1, number_of_best_classifiers + 1):
        scoreSum += sorted_scores[-i]
        a += sorted_scores[-i] * coefficients[indices[-i]][0]
        b += sorted_scores[-i] * coefficients[indices[-i]][1]
    a /= scoreSum
    b /= scoreSum
    return a, b


def get_subspace_limits(j: int, classifier_data: ClassifierData = ClassifierData()):
    """Gets limits of j-th subspace

    :param X: np.array
    :param j: int
    :param classifier_data: ClassifierData
    :return: x_subspace_max, x_subspace_min: float, float
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    minimum = classifier_data.minimum
    maximum = classifier_data.maximum
    x_subspace_min, x_subspace_max = \
        minimum + j * (maximum - minimum) / number_of_space_parts, \
        minimum + (j + 1) * (maximum - minimum) / number_of_space_parts
    return x_subspace_max, x_subspace_min


def test_classifiers(clfs: [], X_validation: [], y_validation: [], X: [], coefficients: [],
                     classifier_data: ClassifierData = ClassifierData()):
    """Tests classifiers

    :param clfs: clfs: [], scikit classifiers
    :param X_validation: np.array
    :param y_validation: np.array
    :param X: np.array
    :param coefficients: []
    :param classifier_data: ClassifierData
    :return: scores, cumulated_scores: [], []
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    write_computed_scores = classifier_data.write_computed_scores
    scores, cumulated_scores, i = [], [], 0
    for clf in clfs:
        score, cumulated_score = [], 0
        for j in range(number_of_space_parts):
            X_part, y_part = prepare_samples_for_subspace(X_validation, y_validation, j, classifier_data)
            if len(X_part) > 0:
                score.append(clf.score(X_part, y_part))
                cumulated_score += clf.score(X_part, y_part) * len(X_part)
            else:
                score.append(0)
        cumulated_score /= len(X_validation)
        scores.append(score)
        cumulated_scores.append(cumulated_score)
        a, b = coefficients[i]

        if write_computed_scores:
            compute_scores_manually(X, X_validation, y_validation, a, b, classifier_data)
        i += 1
    return scores, cumulated_scores


def compute_scores_manually(X: [], X_validation: [], y_validation: [], a: float, b: float,
                            classifier_data: ClassifierData = ClassifierData()):
    """Computes and prints scores manually

    :param X: []
    :param X_validation: []
    :param y_validation: []
    :param a: float
    :param b: float
    :param classifier_data: ClassifierData
    :return:
    """
    print('Computing scores manually')
    number_of_space_parts = classifier_data.number_of_space_parts
    manually_computed_scores, overall_absolute_score = [], 0
    for j in range(number_of_space_parts):
        X_part, y_part = prepare_samples_for_subspace(X_validation, y_validation, j, classifier_data)
        propperly_classified, all_classified = 0, 0
        for k in range(len(X_part)):
            if (a * X_part[k][0] + b > X_part[k][1]) ^ (y_part[k] == 1):
                propperly_classified += 1
            all_classified += 1
        if not (all_classified == 0):
            manually_computed_scores.append(propperly_classified / all_classified)
            overall_absolute_score += propperly_classified
        else:
            manually_computed_scores.append('No samples')
    if 2 * overall_absolute_score < len(X_validation):
        for computed_score in manually_computed_scores:
            try:
                print(1 - computed_score)
            except TypeError:
                print(computed_score)
    else:
        for computed_score in manually_computed_scores:
            print(computed_score)


def compute_confusion_matrix(clfs: [], X_test: [], y_test: []):
    """Calculates confusion matrices for defined classifiers

    :param clfs: []
    :param X_test: np.array
    :param y_test: np.array
    :return: []
    """
    confusion_matrices = []
    for clf in clfs:
        y_predicted = clf.predict(X_test)
        conf_mat = confusion_matrix(y_test, y_predicted)
        confusion_matrices.append(conf_mat)
    return confusion_matrices


def prepare_majority_voting(clfs: [], X_test: [], y_test: [], classifier_data: ClassifierData = ClassifierData()):
    """Returns confusion matrix and score of majority voting of give classifiers

    :param clfs: []
    :param X_test: np.array
    :param y_test: np.array
    :param classifier_data: ClassifierData
    :return: conf_mat, score: [], float
    """
    y_predicted = np.empty(len(X_test), dtype = float)
    for clf in clfs:
        y_predicted += clf.predict(X_test)
    y_predicted /= len(clfs)
    prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1, score = 0, 0, 0, 0, 0
    for i in range(len(y_predicted)):
        if y_test[i] == 0:
            if y_predicted[i] < .5:
                prop_0_pred_0 += 1
            else:
                prop_0_pred_1 += 1
        else:
            if y_predicted[i] < .5:
                prop_1_pred_0 += 1
            else:
                prop_1_pred_1 += 1
    conf_mat = compose_conf_matrix(prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1)
    score = (prop_0_pred_0 + prop_1_pred_1) / len(y_test)
    return np.array(conf_mat), score


def compute_mccs(conf_matrices: []):
    """Computes Matthews correlation coefficient

    :param conf_matrices: []
    :return: mcc: []
    """
    mcc = np.zeros(len(conf_matrices), dtype = float)
    for i in range(len(conf_matrices)):
        prop_0_pred_0, prop_0_pred_1 = conf_matrices[i][0]
        prop_1_pred_0, prop_1_pred_1 = conf_matrices[i][1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mcc_score = compute_mcc(prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1)
        if math.isnan(mcc_score):
            mcc_score = 0
        mcc[i] = mcc_score
    return mcc


def compute_mcc(prop_0_pred_0: int, prop_0_pred_1: int, prop_1_pred_0: int, prop_1_pred_1: int):
    """Computes Matthews correlation coefficient given cells of konfusion matrix

    :param prop_0_pred_0: int
    :param prop_0_pred_1: int
    :param prop_1_pred_0: int
    :param prop_1_pred_1: int
    :return: mcc_score: float
    """
    numerator = prop_0_pred_0 * prop_1_pred_1 - prop_0_pred_1 * prop_1_pred_0
    denominator_sq = (prop_1_pred_1 + prop_0_pred_1) * (prop_1_pred_1 + prop_1_pred_0) * \
                     (prop_0_pred_0 + prop_0_pred_1) * (prop_0_pred_0 + prop_1_pred_0)
    denominator = np.sqrt(denominator_sq)
    mcc_score = numerator / denominator
    return mcc_score


def prepare_composite_mean_classifier(X_test: [], y_test: [], X: [], coefficients: [], scores: [],
                                      number_of_subplots: int,
                                      classifier_data: ClassifierData = ClassifierData()):
    """Prepares composite classifiers using mean strategy

    :param X_test: np.array
    :param y_test: np.array
    :param X: np.array
    :param coefficients: []
    :param scores: []
    :param number_of_subplots: int
    :param classifier_data: ClassifierData
    :return: scores: []
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    show_plots = classifier_data.show_plots
    print('Preparing composite classifier')

    if show_plots:
        ax = plt.subplot(1, number_of_subplots, number_of_subplots)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)

    score, part_lengths, flip_index = [], [], 0
    prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1 = 0, 0, 0, 0
    for j in range(number_of_space_parts):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a, b = evaluate_weighted_average_coefficients_from_n_best(coefficients, scores, j, classifier_data)

        if show_plots:
            x_subspace_min, x_subspace_max = get_subspace_limits(j, classifier_data)
            x = np.linspace(x_subspace_min, x_subspace_max)
            y = a * x + b
            ax.plot(x, y)

        X_part, y_part = prepare_samples_for_subspace(X_test, y_test, j, classifier_data)
        all_classified, propperly_classified = 0, 0
        if len(X_part) > 0 and not (math.isnan(a)) and not (math.isnan(b)):
            for k in range(len(X_part)):
                all_classified += 1
                if y_part[k] >= .5:
                    if a * X_part[k][0] + b > X_part[k][1]:
                        prop_1_pred_1 += 1
                        propperly_classified += 1
                    else:
                        prop_1_pred_0 += 1
                else:
                    if a * X_part[k][0] + b > X_part[k][1]:
                        prop_0_pred_1 += 1
                    else:
                        prop_0_pred_0 += 1
                        propperly_classified += 1
            score.append(propperly_classified / all_classified)
        else:
            score.append(0)
        part_lengths.append(len(X_part))
    cumulated_score = (prop_0_pred_0 + prop_1_pred_1) / (prop_0_pred_0 + prop_0_pred_1 + prop_1_pred_0 + prop_1_pred_1)
    if cumulated_score < 0.5:
        cumulated_score = 1 - cumulated_score
        for i in range(len(score)):
            score[i] = 1 - score[i]
        prop_0_pred_0, prop_0_pred_1 = prop_0_pred_1, prop_0_pred_0
        prop_1_pred_0, prop_1_pred_1 = prop_1_pred_1, prop_1_pred_0
    scores.append(score)
    conf_mat = compose_conf_matrix(prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1)

    if show_plots:
        xx, yy, x_min_plot, x_max_plot, y_min_plot, y_max_plot = get_plot_data(X)
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)

    return scores, cumulated_score, np.array(conf_mat)


def prepare_composite_median_classifier(X_test: [], y_test: [], X: [], coefficients: [], scores: [],
                                        number_of_subplots: int,
                                        classifier_data: ClassifierData = ClassifierData()):
    """Prepares composite classifiers using median strategy

    :param X_test: np.array
    :param y_test: np.array
    :param X: np.array
    :param coefficients: []
    :param scores: []
    :param number_of_subplots: int
    :param classifier_data: ClassifierData
    :return: scores: []
    """
    number_of_space_parts = classifier_data.number_of_space_parts
    show_plots = classifier_data.show_plots
    print('Preparing composite classifier')

    if show_plots:
        ax = plt.subplot(1, number_of_subplots, number_of_subplots)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)

    score, part_lengths, flip_index = [], [], 0
    prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1 = 0, 0, 0, 0
    for j in range(number_of_space_parts):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            filtered_coeffs = reduce_coefficients_in_subspace(coefficients, scores, j, classifier_data)
            is_nan = contain_nan(filtered_coeffs)

        if show_plots:
            x_subspace_min, x_subspace_max = get_subspace_limits(j, classifier_data)
            x = np.linspace(x_subspace_min, x_subspace_max)
            y = np.zeros(shape = (1, len(x)), dtype = float)
            for i in range(len(x)):
                y[i] = get_decision_limit(x[i], filtered_coeffs)
            ax.plot(x, y)

        X_part, y_part = prepare_samples_for_subspace(X_test, y_test, j, classifier_data)
        all_classified, propperly_classified = 0, 0
        if len(X_part) > 0 and not is_nan:
            for k in range(len(X_part)):
                decision_limit = get_decision_limit(X_part[k][0], filtered_coeffs)
                all_classified += 1
                if y_part[k] >= .5:
                    if decision_limit > X_part[k][1]:
                        prop_1_pred_1 += 1
                        propperly_classified += 1
                    else:
                        prop_1_pred_0 += 1
                else:
                    if decision_limit > X_part[k][1]:
                        prop_0_pred_1 += 1
                    else:
                        prop_0_pred_0 += 1
                        propperly_classified += 1
            score.append(propperly_classified / all_classified)
        else:
            score.append(0)
        part_lengths.append(len(X_part))
    cumulated_score = (prop_0_pred_0 + prop_1_pred_1) / (prop_0_pred_0 + prop_0_pred_1 + prop_1_pred_0 + prop_1_pred_1)
    if cumulated_score < 0.5:
        cumulated_score = 1 - cumulated_score
        for i in range(len(score)):
            score[i] = 1 - score[i]
        prop_0_pred_0, prop_0_pred_1 = prop_0_pred_1, prop_0_pred_0
        prop_1_pred_0, prop_1_pred_1 = prop_1_pred_1, prop_1_pred_0
    scores.append(score)
    conf_mat = compose_conf_matrix(prop_0_pred_0, prop_0_pred_1, prop_1_pred_0, prop_1_pred_1)
    if show_plots:
        xx, yy, x_min_plot, x_max_plot, y_min_plot, y_max_plot = get_plot_data(X)
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)
    return scores, cumulated_score, np.array(conf_mat)


def get_decision_limit(sample: float, filtered_coeffs: []):
    """Gets value of decision limit function for given attribute value

    :param sample: float
    :param filtered_coeffs: []
    :return:
    """
    representations = []
    for coeffs in filtered_coeffs:
        representations.append(coeffs[0] * sample + coeffs[1])
    representations = np.sort(representations)
    if len(representations % 2 == 1):
        decision_limit = representations[int((len(representations) + 1) / 2)]
    else:
        decision_limit = representations[int(len(representations) / 2)]
    return decision_limit


def reduce_coefficients_in_subspace(coefficients: [], scores: [], j: int,
                                    classifier_data: ClassifierData = ClassifierData()):
    """Returns array of coefficients for classificator integration (only best)

    :param coefficients: []
    :param scores:[]
    :param j: int
    :param classifier_data: ClassifierData
    :return:
    """
    number_of_best_classifiers = classifier_data.number_of_best_classifiers
    number_of_classifiers = classifier_data.number_of_classifiers
    scores_in_subspace = np.zeros(number_of_classifiers, dtype = float)
    for i in range(len(scores)):
        scores_in_subspace[i] = scores[i][j]
    indices = list(range(number_of_classifiers))
    sorted_scores, indices = (list(t) for t in zip(*sorted(zip(scores_in_subspace, indices))))
    filtered_coeffs = []
    for i in range(1, number_of_best_classifiers + 1):
        filtered_coeffs.append(coefficients[indices[-i]])
    return filtered_coeffs


def contain_nan(filtered_coeffs: []):
    """Checks if coefficient array contains nan

    :param filtered_coeffs: []
    :return:
    """
    contains_nan = False
    for i in range(len(filtered_coeffs)):
        for j in range(len(filtered_coeffs[i])):
            contains_nan = contains_nan or math.isnan(filtered_coeffs[i][j])
    return contains_nan


def compose_conf_matrix(prop_0_pred_0: int, prop_0_pred_1: int, prop_1_pred_0: int, prop_1_pred_1: int):
    """Composes confusion matrix from cells

    :param prop_0_pred_0: int
    :param prop_0_pred_1: int
    :param prop_1_pred_0: int
    :param prop_1_pred_1: int
    :return: conf_mat: []
    """
    prop_0, prop_1 = [], []
    prop_0.append(prop_0_pred_0)
    prop_0.append(prop_0_pred_1)
    prop_1.append(prop_1_pred_0)
    prop_1.append(prop_1_pred_1)
    conf_mat = [prop_0, prop_1]
    return conf_mat


def get_number_of_samples_in_subspace(X: [], j: int, classifier_data: ClassifierData = ClassifierData()):
    """Returns number of samples in j-th subspace

    :param X: np.array
    :param j: int
    :param classifier_data: ClassifierData
    :return: count: int
    """
    x_subspace_max, x_subspace_min = get_subspace_limits(j, classifier_data)
    count = 0
    for i in range(len(X)):
        if x_subspace_min <= X[i][0] <= x_subspace_max:
            count += 1
    return count


def print_scores_pro_classif(scores: []):
    """Prints final scores

    :param scores: []
    :return: void
    """
    i = 1
    for row in scores:
        print('Classifier ' + str(i))
        print(row)
        i += 1


def print_scores_pro_classif_pro_subspace(scores: [], cumulated_scores: []):
    """Prints partial and overall scores

    :param scores: []
    :param cumulated_scores: []
    :return: void
    """
    for i in range(len(scores)):
        print('Scores for {}. classifier'.format(i + 1))
        print(scores[i])
        print('Overall result: {}'.format(cumulated_scores[i]))


def print_scores_conf_mats_pro_classif_pro_subspace(scores: [], cumulated_scores: [], conf_mat: []):
    """Prints partial and overall scores and confusion matrices

    :param scores: []
    :param cumulated_scores: []
    :param conf_mat: []
    :return: void
    """
    print('\n\nIteration results_pro_division\n\n')
    for i in range(len(cumulated_scores)):
        if i == len(cumulated_scores) - 1:
            print('Scores for composite classifier')
            print(scores[i - 1])
            print('Overall result: {}'.format(cumulated_scores[i]))
            for j in range(len(conf_mat[i])):
                print(conf_mat[i][j])
            continue
        if i == len(cumulated_scores) - 2:
            print('Overall result for majority voting: {}'.format(cumulated_scores[i]))
            for j in range(len(conf_mat[i])):
                print(conf_mat[i][j])
            continue
        print('Scores for {}. classifier'.format(i + 1))
        print(scores[i])
        print('Overall result: {}'.format(cumulated_scores[i]))
        for j in range(len(conf_mat[i])):
            print(conf_mat[i][j])


def print_scores_conf_mats_mcc_pro_classif_pro_subspace(scores: [], cumulated_scores: [], conf_mat: [], mcc: []):
    """Prints partial and overall scores, confusion matrices and Matthews correlation coefficients

    :param scores: []
    :param cumulated_scores: []
    :param conf_mat: []
    :param mcc: []
    :return: void
    """
    print('\n\nIteration results_pro_division\n\n')
    for i in range(len(cumulated_scores)):
        if i == len(cumulated_scores) - 1:
            print('Scores for composite classifier')
            print(scores[i - 1])
            print('Overall result: {}'.format(cumulated_scores[i]))
            print('Matthews correlation coefficient: {}'.format(mcc[i]))
            for j in range(len(conf_mat[i])):
                print(conf_mat[i][j])
            continue
        if i == len(cumulated_scores) - 2:
            print('Overall result for majority voting: {}'.format(cumulated_scores[i]))
            print('Matthews correlation coefficient: {}'.format(mcc[i]))
            for j in range(len(conf_mat[i])):
                print(conf_mat[i][j])
            continue
        print('Scores for {}. classifier'.format(i + 1))
        print(scores[i])
        print('Overall result: {}'.format(cumulated_scores[i]))
        print('Matthews correlation coefficient: {}'.format(mcc[i]))
        for j in range(len(conf_mat[i])):
            print(conf_mat[i][j])


def initialize_list_of_lists(numel: int = 1):
    """Creates list of independent lists

    :param numel: int
    :return: []
    """
    return [[] for _ in range(numel)]


def generate_permutations(classifier_data: ClassifierData = ClassifierData()):
    """Generates permutations for datasets

    :param classifier_data: ClassifierData
    :return: []
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    generate_all_permutations = classifier_data.generate_all_permutations
    if generate_all_permutations:
        return itertools.permutations(range(number_of_classifiers + 2), 2)
    else:
        return [(0, 1)]


def get_permutation(X_splitted: [], y_splitted: [], tup: tuple,
                    classifier_data: ClassifierData = ClassifierData()):
    """Returns permutation of datasets, which are moved right

    :param X_splitted: []
    :param y_splitted: []
    :param tup: ()
    :param classifier_data: ClassifierData
    :return: X_whole_train_new, y_whole_train_new, X_validation_new, y_validation_new, X_test_new, y_test_new: [], [],
    np.array, np.array, np.array, np.array
    """
    number_of_classifiers = classifier_data.number_of_classifiers
    X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = [], [], [], [], [], []
    for i in range(number_of_classifiers + 2):
        if i == tup[0]:
            X_validation = X_splitted[i]
            y_validation = y_splitted[i]
        elif i == tup[1]:
            X_test = X_splitted[i]
            y_test = y_splitted[i]
        else:
            X_whole_train.append(X_splitted[i])
            y_whole_train.append(y_splitted[i])
    return X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test


def get_permutation_results(score_pro_permutation: [], mccs_pro_permutation: []):
    """Returns scores and Matthews correlation coefficients after all permutations

    :param score_pro_permutation: []
    :param mccs_pro_permutation: []
    :return: classifier_scores, classifier_mcc: [], []
    """
    classifier_scores = np.mean(score_pro_permutation, axis = 0)
    classifier_mcc = np.mean(mccs_pro_permutation, axis = 0)
    return classifier_scores, classifier_mcc


def get_permutation_stds(score_pro_permutation: [], mccs_pro_permutation: []):
    """Returns scores and Matthews correlation coefficients standars deviations after all permutations

    :param score_pro_permutation: []
    :param mccs_pro_permutation: []
    :return: void
    """
    score_stds = np.std(score_pro_permutation, axis = 0)
    mcc_stds = np.std(mccs_pro_permutation, axis = 0)
    return score_stds, mcc_stds


def print_permutation_results(classifier_scores: [], classifier_mcc: []):
    """Prints overall rsults

    :param classifier_scores: []
    :param classifier_mcc: []
    :return:
    """
    for i in range(len(classifier_scores)):
        if i == len(classifier_scores) - 1:
            print('Composite classifier')
        elif i == len(classifier_scores) - 2:
            print('Majority voting classifier')
        else:
            print('{}. classifier'.format(i))
        print('Score: {}'.format(classifier_scores[i]))
        print('Matthews correlation coefficient: {}'.format(classifier_mcc[i]))


def prepare_result_object(scores: [], mccs: [], scores_stds: [], mccs_stds: []):
    """Prepares result object

    :param scores: []
    :param mccs: []
    :param scores_stds: []
    :param mccs_stds: []
    :return: res: IntegrRes
    """
    mv_score = scores[-2]
    mv_score_std = scores_stds[-2]
    mv_mcc = mccs[-2]
    mv_mcc_std = mccs_stds[-2]
    i_score = scores[-1]
    i_score_std = scores_stds[-1]
    i_mcc = mccs[-1]
    i_mcc_std = mccs_stds[-1]
    res = IntegrRes(mv_score = mv_score,
                    mv_score_std = mv_score_std,
                    mv_mcc = mv_mcc,
                    mv_mcc_std = mv_mcc_std,
                    i_score = i_score,
                    i_score_std = i_score_std,
                    i_mcc = i_mcc,
                    i_mcc_std = i_mcc_std)
    return res


def get_mean_res(partial_ress: []):
    """Prepares mean result object from partials

    :param partial_ress: []
    :return: res: IntegrRes
    """
    mv_score, mv_mcc, i_score, i_mcc = [], [], [], []
    list_of_results = []
    for i in range(len(partial_ress[0])):
        for bagging_result in partial_ress:
            mv_score.append(bagging_result[i].mv_score)
            mv_mcc.append(bagging_result[i].mv_mcc)
            i_score.append(bagging_result[i].i_score)
            i_mcc.append(bagging_result[i].i_mcc)
        res = IntegrRes(mv_score = np.mean(mv_score, axis = 0),
                        mv_score_std = np.std(mv_score, axis = 0),
                        mv_mcc = np.mean(mv_mcc, axis = 0),
                        mv_mcc_std = np.std(mv_mcc, axis = 0),
                        i_score = np.mean(i_score, axis = 0),
                        i_score_std = np.std(i_score, axis = 0),
                        i_mcc = np.mean(i_mcc, axis = 0),
                        i_mcc_std = np.std(i_mcc, axis = 0))
        list_of_results.append(res)
    return list_of_results
