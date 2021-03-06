import os

from DynamicDtreeRes import DynamicDtreeRes
from MathUtils import round_to_str
from nonparametric_tests import friedman_test, bonferroni_dunn_test

import numpy as np

filenames = [
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
      "ye"]
files_num = len(filenames)
references = ['mv', 'rf', 'i']
n_clfs = [3]  # [3, 5, 7, 9]
n_meas = 4
metrics = ['euclidean']
mappings = ['???']


def read(n_clf, metric, mapping):
    name_pattern = 'dynamic-dtree/{}_{}_{}'
    res_filename = name_pattern.format(n_clf, metric, mapping)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    objects = np.empty((files_num, 1), dtype = object)
    with(open(absolute_path)) as file:
        for counter, line in enumerate(file.readlines()):
            values = line.split(',')
            obj = DynamicDtreeRes(
                float(values[0]), float(values[1]), float(values[2]), float(values[3]),  # mv
                float(values[4]), float(values[5]), float(values[6]), float(values[7]),  # rf
                float(values[8]), float(values[9]), float(values[10]), float(values[11]),  # i
                n_clf, metric, mapping, filenames[counter - 1])
            objects[counter] = obj
    return objects


def read_all(n_clf):
    for metric in metrics:
        for mapping in mappings:
            yield read(n_clf, metric, mapping)


def compose_dict(values, aggregate, even=True, vol=True):
    bias = 6 if vol else (8 + len(n_divs) * n_meas * (n_displacements ** n_feas))
    return {
        n_div: aggregate(np.take(np.array(values[i * n_meas * (n_displacements ** n_feas) + bias:(i + 1) * n_meas * (
                n_displacements ** n_feas) + bias], dtype=float), even_indices if even else odd_indices))
        for i, n_div in enumerate(n_divs)}


def map_dtrex(objects, attr, n_clf):
    res_out = []
    all_objs = read_all(n_clf)
    for reference in references:
        res_out.append([getattr(obj, reference + '_' + attr) for obj in all_objs])
    for div in n_divs:
        res_out.append([getattr(obj, 'i_' + method + '_' + attr)[div] for obj in objects])
    return res_out


def create_rank_dict(rankings):
    dict = {}
    for i in range(len(rankings)):
        dict[str(i)] = rankings[i]
    return dict


def custom_print(text, file=None):
    if file is None:
        print(text, end='')
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


def print_results(file_to_write=None):
    for meas in ['acc', 'mcc', 'f1', 'aur']:
        for mapping in ['???']:
            for metric in ['euclidean']:
                for n_clf in n_clfs:
                    custom_print('\nn_clf: ' + str(n_clf) +
                                 ', metric: ' + metric +
                                 ', mapping: ' + mapping +
                                 ', meas: ' + meas + '\n',
                                 file_to_write)

                    for filename in filenames:
                        custom_print(',' + filename, file_to_write)
                    custom_print(',rank\n', file_to_write)

                    objs = read(n_clf, metric, mapping)  # metric & mapping wspólne
                    values = map_dtrex(objs, meas, n_clf)
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
                            custom_print(round_to_str(getattr(obj, 'i_' + mapping + '_' + meas)[div], 3) + ',',
                                         file_to_write)
                        custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                        counter = counter + 1

                    ## post-hoc
                    rankings = create_rank_dict(rankings_cmp)
                    comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
                    pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                    custom_print('p-values: ' + str(pH) + '\n', file_to_write)


with open('reports/3-dynamic-dtree.csv', 'w') as f:
    print_results(f)
