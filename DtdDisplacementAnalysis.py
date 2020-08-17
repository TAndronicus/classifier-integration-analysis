import os

import numpy as np

from DtdDisplacementRes import DtdDisplacementRes
from MathUtils import round_to_str
from nonparametric_tests import friedman_test, bonferroni_dunn_test

filenames = ['bio', 'bup', 'cry', 'dba', 'hab', 'ion', 'met', 'pop', 'sei', 'wdb', 'wis']
# references = ['mv', 'rf', 'wmv_vol', 'wmv_inv']
references = ['mv', 'rf']
n_clfs = [3, 5, 7, 9]
n_feas = 2
n_divs = [20, 40, 60]
n_displacements = 5
n_meas = 2


even_indices = np.arange(0, n_displacements ** n_feas) * n_meas
odd_indices = np.arange(0, n_displacements ** n_feas) * n_meas + 1


def read(n_clf, n_fea):
    name_pattern = 'dtd-displacement/{}_{}_[' + '_'.join([str(el) for el in n_divs]) + ']_{}'
    res_filename = name_pattern.format(n_clf, n_fea, n_displacements)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    objects = []
    with(open(absolute_path)) as file:
        counter = 0
        for line in file.readlines():
            if counter != 0:  # header
                values = line.split(',')
                np.take(np.array(values[6:n_meas * (n_displacements ** n_feas) + 6], dtype = float), even_indices) # acc div20
                obj = DtdDisplacementRes(float(values[0]), float(values[1]),  # mv
                                         float(values[2]), float(values[3]),  # rf
                                         float(values[4]), float(values[5]),  # wmv_vol
                                         compose_dict(values, np.average),
                                         compose_dict(values, np.std),
                                         compose_dict(values, np.average, even = False),
                                         compose_dict(values, np.std, even = False),
                                         float(values[12]), float(values[13]),  # wmv_inv
                                         compose_dict(values, np.average, vol = False),
                                         compose_dict(values, np.std, vol = False),
                                         compose_dict(values, np.average, False, False),
                                         compose_dict(values, np.std, False, False),
                                         n_clf, n_fea, n_divs, filenames[counter - 1])
                objects.append(obj)
            counter += 1
    return objects


def compose_dict(values, aggregate, even = True, vol = True):
    bias = 6 if vol else (8 + len(n_divs) * n_meas * (n_displacements ** n_feas))
    return {
        n_div: aggregate(np.take(np.array(values[i * n_meas * (n_displacements ** n_feas) + bias:(i + 1) * n_meas * (n_displacements ** n_feas) + bias], dtype = float), even_indices if even else odd_indices))
        for i, n_div in enumerate(n_divs)}


def map_dtrex(objects, attr, method):
    res_out = []
    for reference in references:
        res_out.append([getattr(obj, reference + '_' + attr) for obj in objects])
    for div in n_divs:
        res_out.append([getattr(obj, 'i_' + method + '_' + attr)[div] for obj in objects])
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


def print_results(file_to_write = None):
    for meas in ['acc', 'mcc']:
        for mapping in ['vol', 'inv']:
            for n_clf in n_clfs:
                custom_print('\nn_fea: ' + str(n_feas) + ', meas: ' + meas + ', n_clf: ' + str(n_clf) + ', mapping: ' + mapping + '\n', file_to_write)

                for filename in filenames:
                    custom_print(',' + filename, file_to_write)
                custom_print(',rank\n', file_to_write)

                objs = read(n_clf, n_feas)
                values = map_dtrex(objs, meas, mapping)
                iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)

                counter = 0
                for reference in references:
                    custom_print(single_script_psi(reference) + ',', file_to_write)  # TODO: mapping to latex string
                    for filename in filenames:
                        obj = find_first_by_filename(objs, filename)
                        custom_print(round_to_str(getattr(obj, reference + '_' + meas), 3) + ',', file_to_write)
                    custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                    counter = counter + 1

                for div in n_divs:
                    custom_print(double_script_psi(mapping, str(div)) + ',', file_to_write)
                    for filename in filenames:
                        obj = find_first_by_filename(objs, filename)
                        custom_print(round_to_str(getattr(obj, 'i_' + mapping + '_' + meas)[div], 3) + ',', file_to_write)
                    custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                    counter = counter + 1

                ## post-hoc
                rankings = create_rank_dict(rankings_cmp)
                comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
                pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                custom_print('p-values: ' + str(pH) + '\n', file_to_write)


with open('reports/1-displacement.csv', 'w') as f:
    print_results(f)
