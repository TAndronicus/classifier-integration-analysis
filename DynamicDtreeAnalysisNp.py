import os

from DynamicDtreeRes import DynamicDtreeRes
from MathUtils import round_to_str
from nonparametric_tests import friedman_test, bonferroni_dunn_test

import numpy as np

filenames = np.array([
    "aa",
    "ap",
    "ba",
    "bi",
    "bu",
    "c",
    "d",
    "ec",
    "h",
    "i",
    "ir",
    "m",
    "ma",
    "p",
    "ph",
    "pi",
    "ri",
    "sb",
    "se",
    "t",
    "te",
    "th",
    "ti",
    "wd",
    "wi",
    "wr",
    "ww",
    "ye"])
n_files = len(filenames)
references = ['mv', 'rf', 'i']
measurements = ['acc', 'mcc', 'f1', 'aur']
scores = np.array([ref + meas for ref in references for meas in measurements])
n_score = len(scores)
clfs = [3, 5]  # [3, 5, 7, 9]
n_clfs = len(clfs)
metrics = ['e']
n_metrics = len(metrics)
mappings = ['hbd']
n_mappings = len(mappings)


# [filenames x meas/scores x metrics x mappings x n_clf]
# [28 x 12 x 1 x 1 x 4]


def read(metric, mapping, clf):
    name_pattern = 'dynamic-dtree/{}_{}_{}'
    res_filename = name_pattern.format(clf, metric, mapping)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    objects = np.zeros((n_files, n_score), dtype = float)
    with(open(absolute_path)) as file:
        for counter, line in enumerate(file.readlines()):
            values = line.split(',')
            for score in range(0, n_score):
                objects[counter, score] = float(values[score])
    return objects, filenames, scores


def read_cube():
    res = np.zeros((n_files, n_score, n_metrics, n_mappings, n_clfs))
    for i in range(0, n_metrics):
        for j in range(0, n_mappings):
            for k in range(0, n_clfs):
                res[:, :, i, j, k] = read(metrics[i], mappings[j], clfs[k])[0]
    return res, filenames, scores, metrics, mappings, clfs


def average_cube(cube):
    return np.average(cube, axis = (2, 3)), filenames, scores, clfs


def custom_print(text, file = None):
    if file is None:
        print(text, end = '')
    else:
        file.write(text)


def single_script_psi(subscript: str):
    return '$\Psi_{' + subscript + '}$'


def double_script_psi(subscript: str, superscript: str):
    return '$\Psi_{' + subscript + '}^{' + superscript + '}$'


def print_results(file_to_write = None):
    cube, _, _, _, _, _ = read_cube()
    cube_aggregated, _, _, _ = average_cube(cube)
    for i, meas in enumerate(measurements):
        for j, mapping in enumerate(mappings):
            for k, metric in enumerate(metrics):
                for l, clf in enumerate(clfs):
                    custom_print('\nn_clf: ' + str(clf) +
                                 ', metric: ' + metric +
                                 ', mapping: ' + mapping +
                                 ', meas: ' + meas + '\n',
                                 file_to_write)

                    for filename in filenames:
                        custom_print(',' + filename, file_to_write)
                    custom_print(',rank\n', file_to_write)

                    # TODO: stats
                    # objs = read(clf, metric, mapping)  # metric & mapping wspólne
                    # values = map_dtrex(objs, meas, clf)
                    # iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)

                    for n_ref, reference in enumerate(references[:-1]):
                        custom_print(single_script_psi(reference) + ',', file_to_write)  # TODO: mapping to latex string
                        for n_filename, filename in enumerate(filenames):
                            custom_print(round_to_str(cube_aggregated[n_filename, n_ref * len(measurements) + i, l], 3) + ',', file_to_write)
                        # custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                        custom_print('\n', file_to_write)

                    custom_print(single_script_psi(mapping) + ',', file_to_write)
                    for n_filename, filename in enumerate(filenames):
                        custom_print(round_to_str(cube[n_filename, (len(reference) - 1) * len(measurements) + i, k, j, l], 3) + ',', file_to_write)
                    # custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                    custom_print('\n', file_to_write)

                    ## post-hoc
                    # rankings = create_rank_dict(rankings_cmp)
                    # comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
                    # pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                    # custom_print('p-values: ' + str(pH) + '\n', file_to_write)


with open('reports/3-dynamic-dtree.csv', 'w') as f:
    print_results(f)
