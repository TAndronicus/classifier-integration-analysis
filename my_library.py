from sklearn.model_selection import train_test_split


def divide_samples(X, y, number_of_classifiers):
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


def prepare_samples(X_unsplitted, y_unsplitted, quotient):
    X_train, X_test, y_train, y_test, X_test_whole, y_test_whole = [], [], [], [], [], []
    for X_one, y_one in zip(X_unsplitted, y_unsplitted):
        X_train_part, X_test_part, y_train_part, y_test_part = \
            train_test_split(X_one, y_one, train_size=int(len(X_one) * quotient))
        X_train.append(X_train_part)
        X_test.append(X_test_part)
        X_test_whole[len(X_test_whole):] = X_test_part[:]
        y_train.append(y_train_part)
        y_test.append(y_test_part)
        y_test_whole[len(y_test_whole):] = y_test_part[:]

    return X_train, X_test, X_test_whole, y_train, y_test, y_test_whole


def extract_coefficients(clf):
    a, b = - clf.coef_[0][0] / clf.coef_[0][1], - clf.intercept_[0] / clf.coef_[0][1]
    return a, b


def evaluate_coefficients_from_n_best(coefficients, number_of_classifiers, scores, j, number_of_best_classifiers):
    a, b, sumOfScores, params = 0, 0, 0, []
    for i in range(number_of_classifiers):
        params.append([scores[i][j], coefficients[i]])
    params.sort()
    for i in range(number_of_best_classifiers):
        sumOfScores += params[i][0]
        a += params[i][1][0]
        b += params[i][1][1]
    return a / number_of_best_classifiers, b / number_of_best_classifiers


def evaluate_coefficients_from_n_best_weighted(coefficients, number_of_classifiers, scores, j, number_of_best_classifiers):
    a, b, sumOfScores, params = 0, 0, 0, []
    for i in range(number_of_classifiers):
        params.append([scores[i][j], coefficients[i]])
    params.sort()
    for i in range(number_of_best_classifiers):
        sumOfScores += params[i][0]
        a += params[i][1][0]
        b += params[i][1][1]
    return a / sumOfScores, b / sumOfScores


def initialize_classifiers(number_of_classifiers, classifier):
    clfs = []
    for i in range(number_of_classifiers):
        clfs.append(classifier)
    return clfs


def get_samples_limits(X):
    return X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()


def prepare_samples_for_subspace(X_test, y_test, X, j, partitioning):
    x_samp_max, x_samp_min = get_subspace_limits(X, j, partitioning)
    X_part = [row for row in X_test if x_samp_min < row[0] < x_samp_max]
    y_part = [y_test[k] for k in range(len(y_test)) if x_samp_min < X_test[k][0] < x_samp_max]
    return X_part, y_part


def get_subspace_limits(X, j, partitioning):
    x_subspace_min, x_subspace_max = \
        X[:, 0].min() + j * (X[:, 0].max() - X[:, 0].min()) / partitioning, \
        X[:,0].min() + (j + 1) * (X[:, 0].max() - X[:, 0].min()) / partitioning
    return x_subspace_max, x_subspace_min
