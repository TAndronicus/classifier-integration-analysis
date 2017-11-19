import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.extras import average
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import my_library

number_of_samples = 120
mesh_step_size = .2
partitioning = 5
number_of_classifiers = 3

clfs = my_library.initialize_classifiers(number_of_classifiers, LinearSVC())
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                           n_samples=number_of_samples, class_sep=.7, hypercube=False, flip_y=0, random_state=4)
X = StandardScaler().fit_transform(X)
X_unsplitted, y_unsplitted, X_final_test, y_final_test = my_library.divide_samples(X, y, number_of_classifiers)

scores, coefficients = [], []
Xs_train, Xs_test, Xs_test_whole, ys_train, ys_test, ys_test_whole = \
    my_library.prepare_samples(X_unsplitted, y_unsplitted, 2 / 3)

x_min, x_max, y_min, y_max = my_library.get_samples_limits(X)
x_min_plot, x_max_plot, y_min_plot, y_max_plot = x_min - .5, x_max + .5, y_min - .5, y_max + .5
xx, yy = np.meshgrid(np.arange(x_min_plot, x_max_plot, mesh_step_size), np.arange(y_min_plot, y_max_plot, mesh_step_size))

i = 1
for clf, X_train, X_test, y_train, y_test in zip(clfs, Xs_train, Xs_test, ys_train, ys_test):
    score = []
    clf.fit(X_train, y_train)
    for j in range(partitioning):
        X_part, y_part = my_library.prepare_samples_for_subspace(X_test, y_test, X, j, partitioning)
        if (len(X_part) > 0):
            score.append(clf.score(X_part, y_part))
        else:
            score.append(0)

    scores.append(score)
    a, b = my_library.extract_coefficients(clf)
    coefficients.append([a, b])

    ax = plt.subplot(1, number_of_classifiers + 1, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    x = np.linspace(x_min_plot, x_max_plot)
    y = a * x + b
    ax.plot(x, y)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    i += 1
ax = plt.subplot(1, len(clfs) + 1, i)

score, flip_index = [], 0
for j in range(partitioning):
    x_subspace_min, x_subspace_max = my_library.get_subspace_limits(X, j, partitioning)
    x = np.linspace(x_subspace_min, x_subspace_max)
    my_library.evaluate_coefficients_from_two_best(coefficients, number_of_classifiers, scores, j)
    a, b = my_library.evaluate_coefficients(coefficients, number_of_classifiers, scores, j)
    y = a * x + b
    ax.plot(x, y)
    X_part = [row for row in Xs_test_whole if x_subspace_min < row[0] < x_subspace_max]
    y_part = [ys_test_whole[k] for k in range(len(Xs_test_whole)) if x_subspace_min < Xs_test_whole[k][0] < x_subspace_max]
    if (len(X_part) > 0):
        propperly_classified = 0
        all_classified = 0
        flip_index = 0
        for k in range(len(X_part)):
            if ((a * X_part[k][0] + b > X_part[k][1]) ^ (y_part[k] == 1)):
                propperly_classified += 1
            all_classified += 1
        score.append(propperly_classified / all_classified)
        if (propperly_classified / all_classified < 0.5):
            flip_index += 1
        else:
            flip_index -= 1
    else:
        score.append(0)
if (flip_index > 0):
    for k in range(len(score)):
        score[k] = 1 - score[k]
scores.append(score)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

i = 1
for row in scores:
    print('Classifier ' + str(i))
    print(row)
    i+=1
for row in scores:
    print(average(row))
plt.show()