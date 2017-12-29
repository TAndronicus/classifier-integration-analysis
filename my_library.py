from sklearn.model_selection import train_test_split
import xlrd
import numpy as np
from sklearn.feature_selection import SelectKBest


def initialize_classifiers(number_of_classifiers, classifier):
    """Generates list of classifiers for analysis

    :param number_of_classifiers: Number of classifiers
    :param classifier: type of Sklearn classifier
    :return: List of classifiers
    """
    clfs = []
    for i in range(number_of_classifiers):
        clfs.append(classifier)
    return clfs


def load_samples_from_file(filename):
    """Loads data from file

    :param filename: Name of file
    :return: X, y - samples for classification
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
    """Loads data from file

    :param number: Number of sheet
    :return: X, y - samples for classification
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
        if(int(line[number_of_columns - 1].value) == 0):
            X0.append(row)
        else:
            X1.append(row)
        line_number += 1
    # X = SelectKBest(k = 2).fit_transform(X, y)
    X, y = compose_sorted_parts(X0, X1, sheet.nrows)
    return X, y


def compose_sorted_parts(X0, X1, nrows):
    X, y = np.zeros((nrows, 2)), np.zeros(nrows, dtype=np.int)
    X0, X1 = sort_attributes(X0), sort_attributes(X1)
    for i in range(len(X0)):
        X[i, :], y[i] = X0[i], 0
    for i in range(len(X1)):
        X[len(X0) + i, :], y[len(X0) + i] = X1[i], 1
    return X, y


def sort_attributes(X):
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


def sort_results(X, y):
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


def divide_samples_between_classifiers(X, y, number_of_classifiers):
    """Divides sample into parts for every classifier

    Prepares n = number_of_classifiers + 1 parts X(k) and y(k) with the same count
    and returns [X(1), X(2), ..., X(n - 1)], [y(1), y(2), ..., y(n - 1)], X(n), y(n)
    :param X: raw data
    :param y: raw data
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


def divide_samples_between_training_and_testing(X_unsplitted, y_unsplitted, quotient):
    """Divides sample into parts for ttaining and testing

    :param X_unsplitted: [X(1), X(2), ..., X(n - 1)]
    :param y_unsplitted: [y(1), y(2), ..., y(n - 1)]
    :param quotient:
    :return:
    """
    X_train, X_test, y_train, y_test, X_test_whole, y_test_whole = [], [], [], [], [], []
    for X_one, y_one in zip(X_unsplitted, y_unsplitted):
        X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X_one, y_one, train_size=int(len(X_one) * quotient))
        X_train.append(X_train_part)
        X_test.append(X_test_part)
        y_train.append(y_train_part)
        y_test.append(y_test_part)
    return X_train, X_test, y_train, y_test


def prepare_samples_for_subspace(X_test, y_test, X, j, partitioning):
    """Preparing sample for testing in j-th subspace

    :param X_test:
    :param y_test:
    :param X:
    :param j:
    :param partitioning:
    :return: X_part, y_part
    """
    x_samp_max, x_samp_min = get_subspace_limits(X, j, partitioning)
    X_part = [row for row in X_test if x_samp_min < row[0] < x_samp_max]
    y_part = [y_test[k] for k in range(len(y_test)) if x_samp_min < X_test[k][0] < x_samp_max]
    return X_part, y_part


def get_samples_limits(X):
    """Gets limits of j-th subspace

    :param X:
    :return:
    """
    return X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()


def extract_coefficients(clf):
    """Gets a and b coefficients of the classifier

    :param clf:
    :return: a, b
    """
    a, b = - clf.coef_[0][0] / clf.coef_[0][1], - clf.intercept_[0] / clf.coef_[0][1]
    return a, b


def evaluate_average_coefficients_from_n_best(coefficients, number_of_classifiers,
                                              scores, j, number_of_best_classifiers):
    """Evaluates coefficients from n best classifiers in j-th subspace

    :param coefficients:
    :param number_of_classifiers:
    :param scores:
    :param j:
    :param number_of_best_classifiers:
    :return:
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

    :param X:
    :param j:
    :param number_of_subspaces:
    :return: x_subspace_max, x_subspace_min
    """
    x_subspace_min, x_subspace_max = \
        X[:, 0].min() + j * (X[:, 0].max() - X[:, 0].min()) / number_of_subspaces, \
        X[:, 0].min() + (j + 1) * (X[:, 0].max() - X[:, 0].min()) / number_of_subspaces
    return x_subspace_max, x_subspace_min
