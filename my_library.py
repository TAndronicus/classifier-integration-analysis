from sklearn.model_selection import train_test_split
import xlrd
import numpy as np
from sklearn.feature_selection import SelectKBest
from enum import Enum
from deprecation import deprecated


def determine_clf_type(clf):
    """Determines type of classifier

    :param clf:
    :return: clfType: ClfType
    """
    try:
        clf.coef_
        return ClfType.LINEAR
    except:
        print('No coef_ attribute')
    try:
        clf.centroids_
        return ClfType.MEAN
    except:
        print('No centroids_ attribute')


def initialize_classifiers(number_of_classifiers, classifier):
    """Generates list of classifiers for analysis

    :param number_of_classifiers: Number of classifiers
    :param classifier: type of Sklearn classifier
    :return: List of classifiers: [clf, ..., clf]
    """
    clfs = []
    for i in range(number_of_classifiers):
        clfs.append(classifier)
    return clfs


@deprecated()
def load_samples_from_file(filename):
    """Loads data from file

    Warning: method does not sort data
    Missing sorting and composing
    :param filename: Name of file
    :return: X, y: np.array, np.array - samples for classification
    """
    file = xlrd.open_workbook(filename)
    sheet = file.sheet_by_index(0)
    line_number = 0
    line = sheet.row(line_number)
    number_of_columns = len(line)
    X, y = np.zeros((sheet.nrows, number_of_columns - 1)), np.zeros(sheet.nrows, dtype=np.int)
    while line_number < sheet.nrows - 1:
        line_number += 1
        line = sheet.row(line_number)
        row = []
        for i in range(number_of_columns - 1):
            row.append(float(line[i].value))
        X[line_number - 1, :] = row
        y[line_number - 1] = int(line[number_of_columns - 1].value)
    X = SelectKBest(k = 2).fit_transform(X, y)
    return X, y


def load_samples_from_datasets(number):
    """Loads data from dataset (xlsx file with data)

    :param number: Number of sheet
    :return: X, y: np.array, np.array - samples for classification
    """
    file = xlrd.open_workbook('datasets.xlsx')
    sheet = file.sheet_by_index(number)
    line_number = 0
    line = sheet.row(line_number)
    number_of_columns = len(line)
    X0, X1 = [], []
    while line_number < sheet.nrows:
        line = sheet.row(line_number)
        row = []
        for i in range(2):  # range(number_of_columns - 1):
            row.append(float(line[i].value))
        if int(line[number_of_columns - 1].value) == 0:
            X0.append(row)
        else:
            X1.append(row)
        line_number += 1
    # X = SelectKBest(k = 2).fit_transform(X, y)
    print('Ratio (0:1): {}:{}'.format(len(X0), len(X1)))
    X, y = compose_sorted_parts(X0, X1)
    return X, y


def compose_sorted_parts(X0, X1):
    """Composes classification data from 1 and 0 parts

    :param X0: [], data, where y = 0
    :param X1: [], data, where y = 1
    :return: X, y: np.array, np.array - samples for classification
    """
    X, y = np.zeros((len(X0) + len(X1), 2)), np.zeros(len(X0) + len(X1), dtype=np.int)
    X0, X1 = sort_attributes(X0), sort_attributes(X1)
    for i in range(len(X0)):
        X[i, :], y[i] = X0[i], 0
    for i in range(len(X1)):
        X[len(X0) + i, :], y[len(X0) + i] = X1[i], 1
    return X, y


def sort_attributes(X):
    """Sorts attribute array

    :param X: []
    :return: np.array
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


@deprecated('Does not sort by classes (y is not taken into consideration)')
def sort_results(X, y):
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
    X_result, y_result = np.zeros((length, 2)), np.zeros(length, dtype=np.int)
    for i in range(length):
        X_result[i, :] = X_result_array[i]
        y_result[i] = y_result_array[i]
    return X_result, y_result


def divide_generated_samples(X, y):
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
    return  X0, X1


@deprecated('Does not take sorting into consideration')
def divide_samples_between_classifiers(X, y, number_of_classifiers):
    """Divides sample into parts for every classifier

    Warning: does not take sorting into consideration
    Prepares n = number_of_classifiers + 1 parts X(k) and y(k) with the same count
    and returns [X(1), X(2), ..., X(n - 1)], [y(1), y(2), ..., y(n - 1)], X(n), y(n)
    :param X: np.array, raw data
    :param y: np.array, raw data
    :param number_of_classifiers: total number of classifiers (excluding composite)
    :return: [X(1), X(2), ..., X(n - 1)], [y(1), y(2), ..., y(n - 1)], X(n), y(n)
    """
    number_of_samples = len(X)
    X_whole, y_whole, X_rest, y_rest = [], [], [], []
    X_final_test, X_rest, y_final_test, y_rest = \
        train_test_split(X, y, train_size=int(number_of_samples / (number_of_classifiers + 1)))
    for i in range(number_of_classifiers - 1):
        X_part, X_rest, y_part, y_rest = \
            train_test_split(X_rest, y_rest, train_size=int(number_of_samples / (number_of_classifiers + 1)))
        X_whole.append(X_part)
        y_whole.append(y_part)
    X_whole.append(X_rest)
    y_whole.append(y_rest)
    return X_whole, y_whole, X_final_test, y_final_test


def split_sorted_samples_between_classifiers(X, y, number_of_classifiers):
    """Splits sorted samples between classifiers

    :param X: np.array
    :param y: np.array
    :param number_of_classifiers: int
    :return: X_whole, y_whole, X_final_test, y_final_test: [np.array(1), np.array(2), ..., np.array(number_of_classifiers)],
    [np.array(1), np.array(2), ..., np.array(number_of_classifiers)], np.array, np.array
    """
    length = int(len(X) / (number_of_classifiers + 1))
    X_whole, y_whole, X_final_test, y_final_test = [], [], np.zeros((length, 2)), np.zeros(length, dtype=np.int)
    for i in range(number_of_classifiers):
        X_temp, y_temp = np.zeros((length, 2)), np.zeros(length, dtype=np.int)
        for j in range(length):
            X_temp[j, :] = (X[j * (number_of_classifiers + 1) + i, :])
            y_temp[j] = (y[j * (number_of_classifiers + 1) + i])
        X_whole.append(X_temp)
        y_whole.append(y_temp)
    for i in range(length):
        X_final_test[i, :] = (X[(i + 1) * number_of_classifiers + i, :])
        y_final_test[i] = (y[(i + 1) * number_of_classifiers + i])
    return X_whole, y_whole, X_final_test, y_final_test


@deprecated('Does not take sorting into consideration')
def divide_samples_between_training_and_testing(X_unsplitted, y_unsplitted, quotient):
    """Divides sample into parts for ttaining and testing

    Warning: does not take sorting into consideration
    :param X_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param y_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param quotient: float
    :return: X_train, X_test, y_train, y_test: [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)],
    [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)]
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    for X_one, y_one in zip(X_unsplitted, y_unsplitted):
        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X_one, y_one, train_size=int(len(X_one) * quotient))
        X_train.append(X_train_part)
        X_test.append(X_test_part)
        y_train.append(y_train_part)
        y_test.append(y_test_part)
    return X_train, X_test, y_train, y_test


def split_sorted_samples_between_training_and_testing(X_unsplitted, y_unsplitted, quotient):
    """Splits sorted samples for testing and training

    :param X_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param y_unsplitted: [np.array(1), np.array(2), ..., np.array(n - 1)]
    :param quotient: float
    :return: X_train, X_test, y_train, y_test: [np.array(1), np.array(2), ..., np.array(n - 1)], [np.array(1), np.array(2), ..., np.array(n - 1)],
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


def train_test_sorted_split(X_one, y_one, quotient):
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
    X_train, X_test, y_train, y_test = np.zeros((length * (quotient_freq - 1), 2)), np.zeros((length, 2)), np.zeros(length * (quotient_freq - 1), dtype=np.int), \
                                       np.zeros(length, dtype=np.int)
    counter = 0
    for i in range(length):
        for j in range(quotient_freq - 1):
            X_train[counter, :] = X_one[i * quotient_freq + j, :]
            y_train[counter] = y_one[i * quotient_freq + j]
            counter += 1
        X_test[i, :] = X_one[(i + 1) * quotient_freq - 1]
        y_test[i] = y_one[(i + 1) * quotient_freq - 1]
    return X_train, X_test, y_train, y_test


def prepare_samples_for_subspace(X_test, y_test, X, j, partitioning):
    """Preparing sample for testing in j-th subspace

    :param X_test: np.array
    :param y_test: np.array
    :param X: np.array
    :param j: int
    :param partitioning: int
    :return: X_part, y_part: [], []
    """
    x_samp_max, x_samp_min = get_subspace_limits(X, j, partitioning)
    X_part = [row for row in X_test if x_samp_min < row[0] < x_samp_max]
    y_part = [y_test[k] for k in range(len(y_test)) if x_samp_min < X_test[k][0] < x_samp_max]
    return X_part, y_part


def get_samples_limits(X):
    """Gets limits of j-th subspace

    :param X: np.array
    :return: X_0_min, X_0_max, X_1_min, X_1_max: float, float, float, float
    """
    return X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()


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


def evaluate_average_coefficients_from_n_best(coefficients, number_of_classifiers,
                                              scores, j, number_of_best_classifiers):
    """Evaluates coefficients from n best classifiers in j-th subspace

    :param coefficients: []
    :param number_of_classifiers: int
    :param scores: []
    :param j: int
    :param number_of_best_classifiers: int
    :return: a, b: float, float
    """
    a, b, sumOfScores, params = 0, 0, 0, []
    for i in range(number_of_classifiers):
        params.append([scores[i][j], coefficients[i]])
    params.sort()
    for i in range(number_of_best_classifiers):
        sumOfScores += params[i][0]
        a += params[i][1][0]
        b += params[i][1][1]
    return a / number_of_best_classifiers, b / number_of_best_classifiers


def get_subspace_limits(X, j, number_of_subspaces):
    """Gets limits of j-th subspace

    :param X: np.array
    :param j: int
    :param number_of_subspaces: int
    :return: x_subspace_max, x_subspace_min: float, float
    """
    x_subspace_min, x_subspace_max = \
        X[:, 0].min() + j * (X[:, 0].max() - X[:, 0].min()) / number_of_subspaces, \
        X[:, 0].min() + (j + 1) * (X[:, 0].max() - X[:, 0].min()) / number_of_subspaces
    return x_subspace_max, x_subspace_min


class ClfType(Enum):
    """Defines the type of classifier


    """
    LINEAR = 0
    MEAN = 1