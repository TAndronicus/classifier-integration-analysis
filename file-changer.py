import os

alphas = [".3", "1.0"]
n_clfs = [3, 5, 7, 9]
seriex = ['sim', 'pre', 'post-cv', 'post-tr', 'deep', 'shallow']
n_feas = [2]


def mv(n_clf, n_fea, alpha, series):
    name_pattern = "dt/" + series + "/{}_{}_{}"
    res_filename = name_pattern.format(n_clf, n_fea, alpha)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    if os.path.exists(absolute_path):
        new_filename = res_filename[:-5] + res_filename[-3:]
        new_path = os.path.join(os.path.dirname(__file__), new_filename)
        os.rename(absolute_path, new_path)


for n_clf in n_clfs:
    for n_fea in n_feas:
        for alpha in alphas:
            for series in seriex:
                mv(n_clf, n_fea, alpha, series)
