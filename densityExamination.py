import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.extras import average
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from func import prepare_samples, extract_coefficients, evaluate_coefficients

number_of_samples = 900
mesh_step_size = .2
partitioning = 5

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                           n_samples=number_of_samples, class_sep=.7, hypercube=False, flip_y=0, random_state=4)
X = StandardScaler().fit_transform(X)
clf1, clf2, clf3 = LinearSVC(), LinearSVC(), LinearSVC()
clfs = [clf1, clf2, clf3]
number_of_classifiers = len(clfs)
X1, Xrest, y1, yrest = train_test_split(X, y, train_size=int(number_of_samples / number_of_classifiers),
                                        test_size=int(number_of_samples * 2 / number_of_classifiers))
X2, X3, y2, y3 = train_test_split(Xrest, yrest, train_size=int(number_of_samples / number_of_classifiers),
                                  test_size=int(number_of_samples / number_of_classifiers))
X_whole, y_whole = [X1, X2, X3], [y1, y2, y3]
scores, coefficients = [], []
X_train, X_test, X_test_whole, y_train, y_test, y_test_whole = prepare_samples(X_whole, y_whole, number_of_samples,
                                                                               number_of_classifiers)

i = 1
x_min, x_max, y_min, y_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
x_min_plot, x_max_plot, y_min_plot, y_max_plot = x_min - .5, x_max + .5, y_min - .5, y_max + .5
xx, yy = np.meshgrid(np.arange(x_min_plot, x_max_plot, mesh_step_size), np.arange(y_min_plot, y_max_plot, mesh_step_size))

for clf, X_tr, X_te, y_tr, y_te in zip(clfs, X_train, X_test, y_train, y_test):
    score = []
    clf.fit(X_tr, y_tr)
    for j in range(partitioning):
        x_samp_min, x_samp_max = x_min + j * (x_max - x_min) / partitioning, x_min + (j + 1) * (x_max - x_min) / partitioning
        X_temp = [row for row in X_te if x_samp_min < row[0] < x_samp_max]
        y_temp = [y_te[k] for k in range(len(y_te)) if x_samp_min < X_te[k][0] < x_samp_max]
        if (len(X_temp) > 0):
            score.append(clf.score(X_temp, y_temp))
        else:
            score.append(0)
    scores.append(score)
    ax = plt.subplot(1, len(clfs) + 1, i)
    ax.scatter(X_tr[:,0], X_tr[:,1], c=y_tr)
    a, b = extract_coefficients(clf)
    coefficients.append([a, b])
    x = np.linspace(x_min_plot, x_max_plot)
    y = a * x + b
    ax.plot(x, y)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    i += 1
ax = plt.subplot(1, len(clfs) + 1, i)
score=[]
for j in range(partitioning):
    x_samp_min, x_samp_max = x_min + j * (x_max - x_min) / partitioning, x_min + (j + 1) * (x_max - x_min) / partitioning
    x = np.linspace(x_samp_min, x_samp_max)
    a, b = evaluate_coefficients(coefficients, number_of_classifiers, scores, j)
    y = a * x + b
    ax.plot(x, y)
    X_temp = [row for row in X_test_whole if x_samp_min < row[0] < x_samp_max]
    y_temp = [y_test_whole[k] for k in range(len(X_test_whole)) if x_samp_min < X_test_whole[k][0] < x_samp_max]
    if (len(X_temp) > 0):
        propperly_classified = 0
        all_classified = 0
        flip_index = 0
        for k in range(len(X_temp)):
            if ((a * X_temp[k][0] + b > X_temp[k][1]) ^ (y_temp[k] == 1)):
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

for row in scores:
    print(row)
for row in scores:
    print(average(row))
plt.show()