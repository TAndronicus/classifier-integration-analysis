from sklearn.model_selection import train_test_split


def prepare_samples(X_whole, y_whole, number_of_samples, number_of_classifiers):
    X_train, X_test, y_train, y_test, X_test_whole, y_test_whole = [], [], [], [], [], []
    for X_one, y_one in zip(X_whole, y_whole):
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_one, y_one, train_size=int(
            number_of_samples * 2 / (3 * number_of_classifiers)), test_size=int(
            number_of_samples / (3 * number_of_classifiers)))
        X_train.append(X_train_temp)
        X_test.append(X_test_temp)
        X_test_whole[len(X_test_whole):] = X_test_temp[:]
        y_train.append(y_train_temp)
        y_test.append(y_test_temp)
        y_test_whole[len(y_test_whole):] = y_test_temp[:]

    return X_train, X_test, X_test_whole, y_train, y_test, y_test_whole


def extract_coefficients(clf):
    a, b = - clf.coef_[0][0] / clf.coef_[0][1], - clf.intercept_[0] / clf.coef_[0][1]
    return a, b


def evaluate_coefficients(coefficients, number_of_classifiers, scores, j):
    a, b, sumOfScores = 0, 0, 0
    for k in range(number_of_classifiers):
        a += coefficients[k][0] * scores[k][j]
        b += coefficients[k][1] * scores[k][j]
        sumOfScores += scores[k][j]
    return a / sumOfScores, b / sumOfScores
