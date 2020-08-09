import os

import numpy as np

from FileHelper import float_nan_safe
from MathUtils import round_to_str

series = "pre-filtered"
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
references = ['mv', 'rf']
algorithms = ['mv', 'rf', 'i']
measurements = ['acc', 'precisionMi', 'recallMi', 'fScoreMi', 'precisionM', 'recallM', 'fScoreM']
clfs = [3, 5]
alphas = ["0.0", "0.3", "0.7", "1.0"]
betas1 = ["0.5"]  # TODO: flatten
betas2 = ["0.0"]
gammas1 = ["5.0", "20.0", "5.0", "20.0"]
gammas2 = ["5.0", "5.0", "20.0", "20.0"]
dims = ["clf", "alpha", "series"]
n_filenames = len(filenames)
n_algorithms = len(algorithms)
n_measurements = len(measurements)
n_clfs = len(clfs)
n_alphas = len(alphas)
n_gammas = len(gammas1)


def read_cube():
    name_pattern = "dts/" + series + "/{}_{}_{}_{}_{}_{}"
    res = np.zeros((n_filenames, n_algorithms, n_measurements, n_clfs, n_alphas, n_gammas))
    for l, n_clf in enumerate(clfs):
        for m, alpha in enumerate(alphas):
            for n in range(n_gammas):
                res_filename = name_pattern.format(n_clf, alpha, betas1[0], betas2[0], gammas1[n], gammas2[n])
                absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
                with(open(absolute_path)) as file:
                    for i, line in enumerate(file.readlines()):
                        if i == 0: continue
                        values = line.split(',')
                        assert n_algorithms * n_measurements == len(values)
                        for j in range(n_algorithms):
                            for k in range(n_measurements):
                                res[i - 1, j, k, l, m, n] = float_nan_safe(values[j * n_measurements + k])
    return res, filenames, algorithms, measurements, clfs, alphas, zip(gammas1, gammas2)


def aggregate_cube(cube):
    return np.average(cube, axis = (4, 5)), filenames, algorithms, measurements, clfs


def print_results(file_to_write = None):
    cube, _, _, _, _, _, _ = read_cube()
    aggregated_cube, _, _, _, _ = aggregate_cube(cube)
    for l, n_clf in enumerate(clfs):
        for k, measurement in enumerate(measurements):
            for n in range(n_gammas):
                custom_print('\nn_clf: ' + str(n_clf) + ', meas: ' + measurement + ', gamma1: ' + gammas1[n] + ', gamma2: ' + gammas2[n] + '\n', file_to_write)
                for filename in filenames:
                    custom_print(',' + filename, file_to_write)
                custom_print(',rank\n', file_to_write)

                for m, alpha in enumerate(alphas):
                    custom_print(single_script_psi(str(float(alpha))) + ',', file_to_write)
                    for i in range(n_filenames):
                        custom_print(round_to_str(cube[i, n_algorithms - 1, k, l, m, n], 3) + ',', file_to_write)
                    custom_print('\n', file_to_write)  # TODO: remove
                    # custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write) # TODO: stats

                for j, algorithm in enumerate(algorithms[:-1]):
                    custom_print(single_script_psi(algorithm) + ',', file_to_write)
                    for i in range(n_filenames):
                        custom_print(round_to_str(aggregated_cube[i, j, k, l], 3) + ',', file_to_write)
                    custom_print('\n', file_to_write)
                    # custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)

                ## post-hoc # TODO
                # rankings = create_rank_dict(rankings_cmp)
                # comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
                # pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                # custom_print('p-values: ' + str(pH) + '\n', file_to_write)


def custom_print(text, file = None):
    if file is None:
        print(text, end = '')
    else:
        file.write(text)


def single_script_psi(subscript: str):
    return '$\Psi_{' + subscript + '}$'


with open('reports/1-batch.csv', 'w') as f:
    print_results(f)

cube, _, _, _, _, _, _ = read_cube()
aggregated_cube, _, _, _, _ = aggregate_cube(cube)
