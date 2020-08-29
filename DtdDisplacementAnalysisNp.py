import os

import numpy as np

from MathUtils import round_to_str

filenames = ['bio', 'bup', 'cry', 'dba', 'hab', 'ion', 'met', 'pop', 'sei', 'wdb', 'wis']
algorithms = ['MV', 'RF', 'wMV', 'I']
mappings = ['vol', 'inv']
divs = [20, 40, 60]
measurements = ['acc', 'precision', 'recall', 'fscore', 'specificity', 'auc']
displacements = [5]
clfs = [3, 5, 7, 9]
has_header = True
n_filenames = len(filenames)
n_algorithms_independent = 2
n_algorithms_mapping_dep = 1
n_algorithms_mapping_div_dep = 1
n_algorithms = n_algorithms_independent + n_algorithms_mapping_dep + n_algorithms_mapping_div_dep
n_mappings = len(mappings)
n_divs = len(divs)
n_measurements = len(measurements)
n_displacements = len(displacements)
n_clfs = len(clfs)

num_features = 2
references = ['mv', 'rf']


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_cube():
    res = np.zeros((n_filenames, n_algorithms, n_mappings, n_divs, n_measurements, n_displacements, n_clfs))
    for (o, clf) in enumerate(clfs):
        for (n, displacement) in enumerate(displacements):
            name_pattern = 'dtd-displacement/{}_{}_[' + '_'.join([str(el) for el in divs]) + ']_{}'
            res_filename = name_pattern.format(clf, num_features, displacement)
            absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
            with(open(absolute_path)) as file:
                for (line_num, line) in enumerate(file.readlines()):
                    if has_header and line_num == 0: continue
                    file_index = line_num - [0, 1][has_header]
                    values = line.split(',')
                    assert len(values) == (n_algorithms_independent + n_algorithms_mapping_dep * n_mappings + n_algorithms_mapping_div_dep * n_mappings * n_divs) * n_measurements
                    for (algorithm_variation_index, algorithm_values) in enumerate(chunker(values, n_measurements)):
                        if algorithm_variation_index < n_algorithms_independent:
                            res[file_index, algorithm_variation_index, :, :, :, n, o] = algorithm_values
                        elif algorithm_variation_index < n_algorithms_independent + n_algorithms_mapping_dep * n_mappings:
                            algorithm_index = n_algorithms_independent + int((algorithm_variation_index - n_algorithms_independent) / n_mappings)
                            res[file_index, algorithm_index, algorithm_variation_index - n_algorithms_independent, :, :, n, o] = algorithm_values
                        else:
                            rest = algorithm_variation_index - n_algorithms_independent - n_algorithms_mapping_dep * n_mappings
                            algorithm_index = n_algorithms_independent + n_algorithms_mapping_dep + int(rest / (n_mappings * n_divs))
                            res[file_index, algorithm_index, rest % n_mappings, int(rest / n_mappings), :, n, o] = algorithm_values
    return res


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
    cube = read_cube()
    for (n, displacement) in enumerate(displacements):
        for (m, meas) in enumerate(measurements):
            for (k, mapping) in enumerate(mappings):
                for (o, clf) in enumerate(clfs):
                    custom_print('\nn_fea: ' + str(num_features) + ', meas: ' + meas + ', n_clf: ' + str(clf) + ', mapping: ' + mapping + '\n', file_to_write)

                    for filename in filenames:
                        custom_print(',' + filename, file_to_write)
                    custom_print(',rank\n', file_to_write)

                    #                     objs = read(n_clf, n_feas)
                    #                     values = map_dtrex(objs, meas, mapping)
                    #                     iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)
                    #
                    for (j, reference) in enumerate(algorithms[:n_algorithms_independent + n_algorithms_mapping_dep]):
                        custom_print(single_script_psi(reference) + ',', file_to_write)
                        for i in range(len(filenames)):
                            custom_print(round_to_str(cube[i, j, k, 0, m, n, o], 3) + ',', file_to_write)
                        custom_print('\n', file_to_write)
                        # custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)

                    for (l, div) in enumerate(divs):
                        custom_print(double_script_psi(mapping, str(div)) + ',', file_to_write)
                        for i in range(len(filenames)):
                            custom_print(round_to_str(cube[i, -1, k, l, m, n, o], 3) + ',', file_to_write)
                        custom_print('\n', file_to_write)
                        # custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
#
## post-hoc
# rankings = create_rank_dict(rankings_cmp)
# comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
# pH = [x for _, x in sorted(zip(comparisonsH, pH))]
# custom_print('p-values: ' + str(pH) + '\n', file_to_write)


with open('reports/1-displacement.csv', 'w') as f:
    print_results(f)
cube = read_cube()
