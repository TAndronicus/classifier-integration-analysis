from ClfType import ClfType
import xlrd
import numpy as np
from ClassifierData import ClassifierData
import ClassifLibrary
from NotEnoughSamplesError import NotEnoughSamplesError
from sklearn.model_selection import train_test_split

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
    X0, X1 = ClassifLibrary.read_features(sheet, classifier_data)
    classifier_data.columns = columns
    # X = SelectKBest(k = 2).fit_transform(X, y)
    print('Ratio (0:1): {}:{}'.format(len(X0), len(X1)))
    X0, X1 = ClassifLibrary.sort_attributes(X0), ClassifLibrary.sort_attributes(X1)
    ClassifLibrary.assert_distribution_simplified(X0, X1, classifier_data)
    X, y = ClassifLibrary.compose_sorted_parts(X0, X1)
    return X, y


def load_samples_from_datasets_non_parametrised(classifier_data: ClassifierData = ClassifierData()):
    """Loads data from dataset (xlsx file with data)

    :param classifier_data: ClassifierData
    :return: X, y: np.array, np.array - samples for classification
    """
    number_of_dataset_if_not_generated = classifier_data.number_of_dataset_if_not_generated
    file = xlrd.open_workbook('datasets.xlsx')
    sheet = file.sheet_by_index(number_of_dataset_if_not_generated)
    X0, X1 = ClassifLibrary.read_features(sheet, classifier_data)
    print('Ratio (0:1): {}:{}'.format(len(X0), len(X1)))
    X0, X1 = ClassifLibrary.sort_attributes(X0), ClassifLibrary.sort_attributes(X1)
    ClassifLibrary.assert_distribution_simplified(X0, X1, classifier_data)
    X, y = ClassifLibrary.compose_sorted_parts(X0, X1)
    return X, y


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
            ClassifLibrary.train_test_split(X_one, y_one, train_size = int(len(X_one) * quotient))
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


def get_number_of_samples_in_subspace(X: [], j: int, classifier_data: ClassifierData = ClassifierData()):
    """Returns number of samples in j-th subspace

    :param X: np.array
    :param j: int
    :param classifier_data: ClassifierData
    :return: count: int
    """
    x_subspace_min, x_subspace_max = ClassifLibrary.get_subspace_limits(j, classifier_data)
    count = 0
    for i in range(len(X)):
        if x_subspace_min <= X[i][0] <= x_subspace_max:
            count += 1
    return count