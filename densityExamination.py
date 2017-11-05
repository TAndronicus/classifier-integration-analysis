from numpy.ma.extras import average
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

number_of_samples = 90000
mesh_step_size = .2
partitioning = 5

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=2, n_clusters_per_class=1, n_samples=number_of_samples, class_sep=.7)
X = StandardScaler().fit_transform(X)
clf1, clf2, clf3 = LinearSVC(), LinearSVC(), LinearSVC()
clfs = [clf1, clf2, clf3]
X1, Xrest, y1, yrest = train_test_split(X, y, train_size=int(number_of_samples / 3), test_size=int(number_of_samples * 2 / 3))
X2, X3, y2, y3 = train_test_split(Xrest, yrest, train_size=int(number_of_samples / 3), test_size=int(number_of_samples / 3))
X_whole, y_whole = [X1, X2, X3], [y1, y2, y3]
X_train, X_test, y_train, y_test, scores, coefficients, X_test_whole, y_test_whole = [], [], [], [], [], [], [], []

for X_one, y_one in zip(X_whole, y_whole):
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_one, y_one, train_size=int(number_of_samples * 2 / 9), test_size=int(number_of_samples / 9))
    X_train.append(X_train_temp)
    X_test.append(X_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)
    X_test_whole[len(X_test_whole):] = X_test_temp[:]
    y_test_whole[len(y_test_whole):] = y_test_temp[:]

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
        score.append(clf.score(X_temp, y_temp))
    scores.append(score)
    ax = plt.subplot(1, len(clfs) + 1, i)
    ax.scatter(X_tr[:,0], X_tr[:,1], c=y_tr)
    a, b = - clf.coef_[0][0] / clf.coef_[0][1], - clf.intercept_[0] / clf.coef_[0][1]
    coefficients.append([a, b])
    x = np.linspace(x_min_plot, x_max_plot)
    y = a * x + b
    ax.plot(x, y)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    i += 1
ax = plt.subplot(1, len(clfs) + 1, i)
#for row in X_test:
score=[]
for j in range(partitioning):
    finalClassifier = LinearSVC()
    x_samp_min, x_samp_max = x_min + j * (x_max - x_min) / partitioning, x_min + (j + 1) * (x_max - x_min) / partitioning
    x = np.linspace(x_samp_min, x_samp_max)
    a, b, sumOfScores = 0, 0, 0
    for k in range(len(clfs)):
        a += coefficients[k][0] * scores[k][j]
        b += coefficients[k][1] * scores[k][j]
        sumOfScores += scores[k][j]
    a, b = a / sumOfScores, b / sumOfScores
    y = a * x + b
    ax.plot(x, y)
    X_temp = [row for row in X_test_whole if x_samp_min < row[0] < x_samp_max]
    y_temp = [y_test_whole[k] for k in range(len(X_test_whole)) if x_samp_min < X_test_whole[k][0] < x_samp_max]
    finalClassifier.coef_=[-a, 1]
    finalClassifier.intercept_=[-b]
    score.append(clf.score(X_temp, y_temp))
scores.append(score)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

for row in scores:
    print(row)
for row in scores:
    print(average(row))
plt.show()