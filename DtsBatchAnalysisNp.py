import os

import numpy as np

from DtsBatchRes import DtsBatchRes
from MathUtils import round_to_str
from nonparametric_tests import friedman_test, bonferroni_dunn_test

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
                                res[i - 1, j, k, l, m, n] = values[j * n_measurements + k]
    return res, filenames, algorithms, measurements, clfs, alphas, zip(gammas1, gammas2)


def read(n_clf, alpha, series, gamma_permutation = -1):
    name_pattern = "dts/" + series + "/{}_{}_{}_{}_{}_{}"
    objects = []
    for gamma_number in [range(len(gammas1)), [gamma_permutation]][gamma_permutation != -1]:
        res_filename = name_pattern.format(n_clf, alpha, betas1[0], betas2[0], gammas1[gamma_number], gammas2[gamma_number])
        absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
        with(open(absolute_path)) as file:
            counter = 0
            for line in file.readlines():
                values = line.split(',')
                if len(values) < 6: continue
                obj = DtsBatchRes(float(values[0]), float(values[1]),  # mv
                                  float(values[4]), float(values[5]),  # rf
                                  float(values[8]), float(values[9]),  # i
                                  n_clf,
                                  alpha, betas1[0], betas2[0], gammas1[gamma_number], gammas2[gamma_number],
                                  filenames[counter])
                objects.append(obj)
                counter += 1
    return objects


def get_average_reference(objects, attr, reference):
    res_out = []
    length = len(objects)
    for i in range(len(objects[0])):
        value = 0
        for j in range(length):
            value += getattr(objects[j][i], reference + "_" + attr)
        res_out.append(value / length)
    return res_out


def map_dtrex(objects, attr):
    res_out = []
    for obj_out in objects:
        res_in = []
        for obj_in in obj_out:
            res_in.append(getattr(obj_in, "i_" + attr))
        res_out.append(res_in)
    for reference in references:
        res_out.append(get_average_reference(objects, attr, reference))
    return res_out


def create_rank_dict(rankings):
    dict = {}
    for i in range(len(rankings)):
        dict[str(i)] = rankings[i]
    return dict


def custom_print(text, file = None):
    if file is None:
        print(text, end = '')
    else:
        file.write(text)


def find_first_by_filename(objects, filename):
    for object in objects:
        if object.filename == filename:
            return object
    raise Exception('Filename not found: ' + filename)


def single_script_psi(subscript: str):
    return '$\Psi_{' + subscript + '}$'


def double_script_psi(subscript: str, superscript: str):
    return '$\Psi_{' + subscript + '}^{' + superscript + '}$'


def initialize_sums_by_filenames():
    res = {}
    for filename in filenames:
        res[filename] = 0
    return res


def get_params_suffix(gamma_permutation):
    suffix = ""
    if gamma_permutation != -1:
        suffix += (", gamma1: " + gammas1[gamma_permutation] + ", gamma2: " + gammas2[gamma_permutation])
    return suffix


def print_results(file_to_write = None):
    for n_clf in n_clfs:
        for meas in ['acc', 'mcc']:
            objs = []
            for alpha in alphas:
                objs.append(read(n_clf, alpha, series))
            ref_values = map_dtrex(objs, meas)
            for gamma_permutation in range(len(gammas1)):
                suffix = get_params_suffix(gamma_permutation)
                custom_print('\nn_clf: ' + str(n_clf) + ', meas: ' + meas + suffix + '\n', file_to_write)

                for filename_index in filenames:
                    custom_print(',' + filename_index, file_to_write)
                custom_print(',rank\n', file_to_write)

                filtered_objs = []
                for alpha in alphas:
                    filtered_objs.append(read(n_clf, alpha, series, gamma_permutation))
                values = map_dtrex(filtered_objs, meas)
                iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)  # wrong stats - only means from current gamma permutation are taken

                counter = 0
                for alpha in alphas:
                    custom_print(single_script_psi(str(float(alpha))) + ',', file_to_write)
                    for filename_index in range(len(filenames)):
                        custom_print(round_to_str(values[counter][filename_index], 3) + ',', file_to_write)
                    custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                    counter = counter + 1

                for reference in references:
                    custom_print(single_script_psi(reference) + ',', file_to_write)
                    for filename_index in range(len(filenames)):
                        custom_print(round_to_str(ref_values[counter][filename_index], 3) + ',', file_to_write)
                    custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                    counter = counter + 1

                ## post-hoc
                rankings = create_rank_dict(rankings_cmp)
                comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
                pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                custom_print('p-values: ' + str(pH) + '\n', file_to_write)


def aggregate_csv():
    for n_clf in n_clfs:
        for meas in ['acc', 'mcc']:
            objs = []
            for alpha in alphas:
                objs.append(read(n_clf, alpha, series))
            ref_values = map_dtrex(objs, meas)
            for gamma_permutation in range(len(gammas1)):
                filtered_objs = []
                for alpha in alphas:
                    filtered_objs.append(read(n_clf, alpha, series, gamma_permutation))
                values = map_dtrex(filtered_objs, meas)
                with open('csv/' + meas + '_' + gammas1[gamma_permutation] + '_' + gammas2[gamma_permutation] + '.csv', 'w') as file_to_write:
                    custom_print(','.join(['alpha' + a + '(' + meas + ')' for a in alphas]) + ',mv(' + meas + '),rf(' + meas + ')\n', file_to_write)
                    for i in range(len(values[0])):
                        for j in range(len(values)):
                            custom_print(round_to_str([values, ref_values][j in [4, 5]][j][i], 3) + ',', file_to_write)
                        custom_print('\n', file_to_write)


# with open('reports/1-batch.csv', 'w') as f:
#     print_results(f)

# aggregate_csv()

cube, _, _, _, _, _, _ = read_cube()
