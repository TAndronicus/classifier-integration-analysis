import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import NearestCentroid
import MyLibrary

# clf = NearestCentroid()
clf = LinearSVC(max_iter = 1e8, tol = 1e-10)
are_samples_generated = True
number_of_samples_if_generated = 90
number_of_dataset_if_not_generated = 0
plot_mesh_step_size = .2
number_of_space_parts = 5
number_of_classifiers = 3
number_of_best_classifiers = number_of_classifiers - 1
training2testing_quotient = 2 / 3
draw_color_plot = False
write_computed_scores = False

# Preparing classifiers
print('Prepare classifiers')
clfs = MyLibrary.initialize_classifiers(number_of_classifiers, clf)

# Prepare raw data
print('Prepare raw data')
if are_samples_generated:
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, n_samples=number_of_samples_if_generated, class_sep=2.7, hypercube=False, random_state=2)
    X0, X1 = MyLibrary.divide_generated_samples(X, y)
    X0, X1 = MyLibrary.sort_attributes(X0), MyLibrary.sort_attributes(X1)
    X, y = MyLibrary.compose_sorted_parts(X0, X1)
else:
    X, y = MyLibrary.load_samples_from_datasets(number_of_dataset_if_not_generated)
    # X, y = my_library.load_samples_from_file('Dane_9_12_2017.xlsx')

# Splitting between classifiers
print('Split between classifiers')
X_unsplitted, y_unsplitted, X_final_test, y_final_test = MyLibrary.split_sorted_samples_between_classifiers(X, y, number_of_classifiers)

# Splitting between training and testing
print('Split between training and testing')
scores, coefficients = [], []
Xs_train, Xs_test, ys_train, ys_test = MyLibrary.split_sorted_samples_between_training_and_testing(X_unsplitted, y_unsplitted, training2testing_quotient)

# Getting data for plot
print('Get data for plot')
x_min, x_max, y_min, y_max = MyLibrary.get_samples_limits(X)
x_shift = 0.1 * (x_max - x_min)
y_shift = 0.1 * (y_max - y_min)
x_min_plot, x_max_plot, y_min_plot, y_max_plot = x_min - x_shift, x_max + x_shift, y_min - y_shift, y_max + y_shift
xx, yy = np.meshgrid(np.arange(x_min_plot, x_max_plot, plot_mesh_step_size), np.arange(y_min_plot, y_max_plot, plot_mesh_step_size))
if draw_color_plot:
    number_of_subplots =  number_of_classifiers * 2 + 1
else:
    number_of_subplots = number_of_classifiers + 1

# Training and testing classifiers
print('Train and test classifiers')
i = 1
for clf, X_train, X_test, y_train, y_test in zip(clfs, Xs_train, Xs_test, ys_train, ys_test):
    score = []
    # Training classifier
    clf.fit(X_train, y_train)
    type_of_classifier = MyLibrary.determine_clf_type(clf)
    # Testing classifier for every subspace
    for j in range(number_of_space_parts):
        X_part, y_part = MyLibrary.prepare_samples_for_subspace(X_test, y_test, X, j, number_of_space_parts)
        if len(X_part) > 0:
            score.append(clf.score(X_part, y_part))
        else:
            score.append(0)
    scores.append(score)
    if type_of_classifier == MyLibrary.ClfType.LINEAR:
        a, b = MyLibrary.extract_coefficients_for_linear(clf)
    elif type_of_classifier == MyLibrary.ClfType.MEAN:
        a, b = MyLibrary.extract_coefficients_for_mean(clf)

    if write_computed_scores:
        # Computing scores manually
        print('Compute scores manually')
        for j in range(number_of_space_parts):
            X_part, y_part = MyLibrary.prepare_samples_for_subspace(X_test, y_test, X, j, number_of_space_parts)
            propperly_classified, all_classified = 0, 0
            for k in range(len(X_part)):
                if (a * X_part[k][0] + b > X_part[k][1]) ^ (y_part[k] == 1):
                    propperly_classified += 1
                all_classified += 1
            if not(all_classified == 0):
                print(propperly_classified / all_classified)
            else:
                print('No samples')

    coefficients.append([a, b])
    # Prepare plot
    ax = plt.subplot(1, number_of_subplots, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    x = np.linspace(x_min_plot, x_max_plot)
    y = a * x + b
    ax.plot(x, y)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    i += 1

    if draw_color_plot:
        # Draw color plot
        print('Drawing color plot')
        ax = plt.subplot(1, number_of_subplots, i)
        if hasattr(clf, 'decision_function'):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        elif hasattr(clf, 'predict_proba'):
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha = .8)
        i += 1

# Prepare plot of composite
ax = plt.subplot(1, number_of_subplots, i)
ax.scatter(X_final_test[:, 0], X_final_test[:, 1], c=y_final_test)

# Preparing composite classifier
print('Prepare composite classifier')
score, flip_index = [], 0
for j in range(number_of_space_parts):
    x_subspace_min, x_subspace_max = MyLibrary.get_subspace_limits(X, j, number_of_space_parts)
    x = np.linspace(x_subspace_min, x_subspace_max)
    a, b = MyLibrary.evaluate_average_coefficients_from_n_best(coefficients, number_of_classifiers, scores, j, number_of_best_classifiers)
    y = a * x + b
    ax.plot(x, y)
    X_part, y_part = MyLibrary.prepare_samples_for_subspace(X_final_test, y_final_test, X, j, number_of_space_parts)
    # Checking if the orientation is correct and determining score
    if len(X_part) > 0:
        propperly_classified = 0
        all_classified = 0
        flip_index = 0
        for k in range(len(X_part)):
            if (a * X_part[k][0] + b > X_part[k][1]) ^ (y_part[k] == 1):
                propperly_classified += 1
            all_classified += 1
        if propperly_classified / all_classified < 0.5:
            score.append(1 - propperly_classified / all_classified)
            flip_index += 1
        else:
            score.append(propperly_classified / all_classified)
            flip_index -= 1
    else:
        score.append(0)
if flip_index > 0:
    for k in range(len(score)):
        score[k] = 1 - score[k]
scores.append(score)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

# Print resilts
i = 1
for row in scores:
    print('Classifier ' + str(i))
    print(row)
    i += 1
plt.show()
