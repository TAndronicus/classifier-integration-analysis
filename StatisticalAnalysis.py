from ClassificationAnalysis import read_in_objects
from ClassifLibrary import initialize_list_of_lists
from scipy.stats import kstest, shapiro, friedmanchisquare, f_oneway, kruskal, wilcoxon, ttest_ind, median_test
import numpy as np
import matplotlib.pyplot as plt
from nonparametric_tests import friedman_aligned_ranks_test, friedman_test, bonferroni_dunn_test, holm_test

objects = read_in_objects()
filenames = ['biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx',
             'data_banknote_authentication.csv', 'haberman.dat',
             'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv',
             'seismic_bumps.dat', 'twonorm.dat', 'wdbc.dat',
             'wisconsin.dat']

spaces = [3, 4, 5, 6, 7, 8, 9, 10]
clfs = [3, 4, 5, 6, 7, 8, 9]
best = [2, 3, 4, 5, 6, 7, 8]
method_dict_short = {
    0: "A",
    1: "M"
}
method_dict_long = {
    0: "Weighted average",
    1: "Median"
}
measures_dict = {
    "ia": "i_score",
    "im": "i_mcc",
    "mva": "mv_score",
    "mvm": "mv_mcc"
}
measures = ['score', 'mcc']
intRef = ['i', 'mv']

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


def holm_min(rankings):
    dict = {}
    for i in range(0, len(rankings)):
        dict[str(i)] = rankings[i]
    comparisonsH, z, pH, adj_p = holm_test(dict, str(len(rankings) - 1))
    comparisonsD, z_values, p_values, adj_p_values = bonferroni_dunn_test(dict, str(len(rankings) - 1))
    return min(pH)


def print_p_val_multi_post_hoc(pF, pH):
    print('pF = ' + str(pF) + ', pH = ' + str(pH))


def find_by_filename(objs, filename):
    for obj in objs:
        if obj.filename == filename:
            return obj
    return None


# obj = get_dependent_from_space_parts(5, 3, list(range(3, 11)), 0, 0)
# valsScore = extract_param(obj, 'i_score')
# pF, pH = friedman_holm_test(valsScore)
# print_p_val_multi_post_hoc(pF, pH)
# valsMcc = extract_param(obj, 'i_mcc')
# pF, pH = friedman_holm_test(valsMcc)
# print_p_val_multi_post_hoc(pF, pH)


def extract_mv_param(obj, param):
    row = []
    for o in obj[0]:
        row.append(getattr(o, param))
    return row


def format_to_length(number: float, length: int):
    number_as_int = int(pow(10, length) * number)
    if pow(10, length) * number - number_as_int > .5:
        number_as_int = number_as_int + 1
    number_as_str = str(.0 + number_as_int / pow(10, length))[0:length + 2]
    while len(number_as_str) < length + 2:
        number_as_str = number_as_str + '0'
    return number_as_str

pH = {}
for space in spaces:
    for measure in measures:
        resname = measure + '_' + str(space)
        with open(resname + '.csv', 'w') as file:
    # header (filenames)
            for filename in filenames:
                file.write(',' + filename.split('.')[0][0:3])
            file.write(",Rank\n")
            for (key, val) in method_dict_short.items(): # iteracja po metodach
                ranks = []
                obj = get_dependent_from_n_best(9, best, space, key, 0)
                valsScore = extract_param(obj, 'i_' + measure)
                mv_scores = extract_mv_param(obj, 'mv_' + measure)
                valsScore.append(mv_scores)
                f, p, rankings, pivots = friedman_test(valsScore)
                pH[val + '_' + resname] = holm_min(rankings)
                for best_index in range(0, len(best)):
                    file.write('\clf{' + val + '}{' + str(best[best_index]) + '}')
                    for filename in filenames:
                        f_obj = find_by_filename(obj[best_index], filename)
                        file.write(',' + format_to_length(getattr(f_obj, 'i_' + measure), 3))
                    file.write(',' + format_to_length(rankings[best_index], 2))
                    file.write('\n')
                file.write('\clf{MV}')
                for filename in filenames:
                    f_obj = find_by_filename(get_dependent_from_n_best(9, best, space, 0, 0)[0], filename)
                    file.write(',' + format_to_length(getattr(f_obj, 'mv_' + measure), 3))
                file.write(',' + format_to_length(rankings[-1], 2))
                file.write('\n')


with open('holm.csv', 'w') as file:
    file.write('n_s,A,,M,\n')
    file.write(',ACC,MCC,ACC,MCC\n')
    for space in spaces:
        if space % 2 == 0:
            continue
        file.write(str(space))
        for (_, val) in method_dict_short.items():
            for measure in measures:
                file.write(',' + format_to_length(pH[val + '_' + measure + '_' + str(space)], 2))
        file.write('\n')

with open('stat.csv', 'w') as file:
    file.write('para,z,p_dunn,p_holm\n')
    for space in spaces:
        for measure in measures:
            for (key, val) in method_dict_short.items(): # iteracja po metodach
                file.write('Subspaces: ' + str(space) + '\n')
                file.write('Measure: ' + measure + '\n')
                file.write('Method: ' + val + '\n')
                obj = get_dependent_from_n_best(9, best, space, key, 0)
                valsScore = extract_param(obj, 'i_' + measure)
                mv_scores = extract_mv_param(obj, 'mv_' + measure)
                valsScore.append(mv_scores)
                f, p, rankings, pivots = friedman_test(valsScore)
                dict = {}
                for i in range(0, len(rankings)):
                    dict[str(i)] = rankings[i]
                comparisonsH, z, pH, adj_p = holm_test(dict, str(len(rankings) - 1))
                comparisonsD, z_values, p_values, adj_p_values = bonferroni_dunn_test(dict, str(len(rankings) - 1))
                for i in range(0, len(pH)):
                    file.write(comparisonsD[i] + ',' + format_to_length(z[i], 2) + ',' + format_to_length(p_values[i], 2) + ',' + format_to_length(pH[i], 2) + '\n')

