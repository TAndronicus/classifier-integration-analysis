from ClassificationAnalysis import read_in_objects
from ClassifLibrary import initialize_list_of_lists
from scipy.stats import kstest, shapiro, friedmanchisquare, f_oneway, kruskal, wilcoxon, ttest_ind, median_test
import numpy as np
import matplotlib.pyplot as plt
from nonparametric_tests import friedman_aligned_ranks_test, friedman_test, bonferroni_dunn_test, holm_test

objects = read_in_objects()


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis = 0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis = 0)
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color = 'gray', linestyle = '--')
    plt.axhline(md + 1.96 * sd, color = 'gray', linestyle = '--')
    plt.axhline(md - 1.96 * sd, color = 'gray', linestyle = '--')


def get_dependent_from_n_class_const(n_class, n_best, space_parts, i_meth, bagging):
    subjects = initialize_list_of_lists(len(n_class))
    for i in range(len(n_class)):
        for object in objects:
            if object.n_class == n_class[i] and \
                    object.n_best == n_best and \
                    object.space_parts == space_parts and \
                    object.i_meth == i_meth and \
                    object.bagging == bagging:
                subjects[i].append(object)
    return subjects


def get_mv_i_diff(i_meth, bagging, measure):
    means, diffs = [], []
    for object in objects:
        if object.i_meth == i_meth and \
                object.bagging == bagging:
            if measure == 'score':
                means.append((object.i_score + object.mv_score) / 2)
                diffs.append(object.i_score - object.mv_score)
            else:
                means.append((object.i_mcc + object.mv_mcc) / 2)
                diffs.append(object.i_mcc - object.mv_mcc)
    return means, diffs


def get_diff_meth(bagging):
    mean_score, median_score, mean_mcc, median_mcc = [], [], [], []
    for object in objects:
        if object.bagging == bagging:
            if object.i_meth == 0:
                mean_score.append(object.i_score)
                mean_mcc.append(object.i_mcc)
            else:
                median_score.append(object.i_score)
                median_mcc.append(object.i_mcc)
    return mean_score, median_score, mean_mcc, median_mcc


def get_diff_bag():
    bag_score, nbag_score, bag_mcc, nbag_mcc = [], [], [], []
    for object in objects:
        if object.bagging == 0:
            nbag_score.append(object.i_score)
            nbag_mcc.append(object.i_mcc)
        else:
            bag_score.append(object.i_score)
            bag_mcc.append(object.i_mcc)
    return bag_score, nbag_score, bag_mcc, nbag_mcc


def get_mv_i(i_meth, bagging):
    i_score, mv_score, i_mcc, mv_mcc = [], [], [], []
    for object in objects:
        if object.i_meth == i_meth and \
                object.bagging == bagging:
            i_score.append(object.i_score)
            mv_score.append(object.mv_score)
            i_mcc.append(object.i_mcc)
            mv_mcc.append(object.mv_mcc)
    return i_score, mv_score, i_mcc, mv_mcc


def get_dependent_from_n_class_non_const(n_class, diff, space_parts, i_meth, bagging):
    subjects = initialize_list_of_lists(len(n_class))
    for i in range(len(n_class)):
        for object in objects:
            if object.n_class == n_class[i] and \
                    object.n_best == n_class[i] - diff and \
                    object.space_parts == space_parts and \
                    object.i_meth == i_meth and \
                    object.bagging == bagging:
                subjects[i].append(object)
    return subjects


def get_dependent_from_n_best(n_class, n_best, space_parts, i_meth, bagging):
    subjects = initialize_list_of_lists(len(n_best))
    for i in range(len(n_best)):
        for object in objects:
            if object.n_class == n_class and \
                    object.n_best == n_best[i] and \
                    object.space_parts == space_parts and \
                    object.i_meth == i_meth and \
                    object.bagging == bagging:
                subjects[i].append(object)
    return subjects


def get_dependent_from_space_parts(n_class, n_best, space_parts, i_meth, bagging):
    subjects = initialize_list_of_lists(len(space_parts))
    for i in range(len(space_parts)):
        for object in objects:
            if object.n_class == n_class and \
                    object.n_best == n_best and \
                    object.space_parts == space_parts[i] and \
                    object.i_meth == i_meth and \
                    object.bagging == bagging:
                subjects[i].append(object)
    return subjects


def get_min_stat(subjects, alpha = .005):
    minstat = 1
    for subject in subjects:
        ks = kstest(subject, 'norm')
        sw = shapiro(subject)
        minstat = min(ks[1], sw[1], minstat)
        if minstat < alpha:
            print('Not normal')
            break
    if minstat >= alpha:
        print('Normal')
    return minstat


def get_p_ks(subjects, alpha = .005):
    minstat = 1
    for subject in subjects:
        ks = kstest(subject, 'norm')
        minstat = min(ks[1], minstat)
        if minstat < alpha:
            # print('Not normal')
            break
    if minstat >= alpha:
        # print('Normal')
        pass
    return minstat


def get_p_sw(subjects, alpha = .005):
    minstat = 1
    for subject in subjects:
        sw = shapiro(subject)
        minstat = min(sw[1], minstat)
        if minstat < alpha:
            # print('Not normal')
            break
    if minstat >= alpha:
        # print('Normal')
        pass
    return minstat


def extract_arrtibute(subjects, attr):
    result = []
    for subject in subjects:
        partial_result = []
        for object in subject:
            partial_result.append(getattr(object, attr))
        result.append(partial_result)
    return result


def test_friedman(subjects, alpha = .005):
    w, p = friedmanchisquare(*subjects)
    if p < alpha:
        # print('Different distribution')
        pass
    else:
        # print('Same distribution')
        pass
    return p


def convert_to_numpy(array):
    return np.array(array)


def normalize_variance(subjects):
    means = np.mean(subjects, axis = 1)
    stds = np.std(subjects, axis = 1)
    result = []
    for i in range(len(subjects)):
        result.append(((subjects[i] - means[i]) / stds[i]) + means[i])
    return result


def extract_param(objects, param):
    vals = []
    for o in objects:
        row = []
        for val in o:
            row.append(getattr(val, param))
        vals.append(row)
    return vals


def friedman_holm_test(vals):
    f, p, rankings, pivots = friedman_test(vals)
    dict = {}
    for i in range(0, len(rankings)):
        dict[str(i)] = rankings[i]
    pvals = []
    for i in range(0, len(rankings)):
        comparisons, z, pH, adj_p = holm_test(dict, str(i))
        pvals.append(min(adj_p))
    return p, min(pvals)


def print_p_val_multi_post_hoc(pF, pH):
    print('pF = ' + str(pF) + ', pH = ' + str(pH))


obj = get_dependent_from_space_parts(5, 3, list(range(3, 11)), 0, 0)
valsScore = extract_param(obj, 'i_score')
pF, pH = friedman_holm_test(valsScore)
print_p_val_multi_post_hoc(pF, pH)
valsMcc = extract_param(obj, 'i_mcc')
pF, pH = friedman_holm_test(valsMcc)
print_p_val_multi_post_hoc(pF, pH)
