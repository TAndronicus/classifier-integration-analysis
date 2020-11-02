import os
from math import pow, sqrt

import numpy as np
from texttable import Texttable

filenames = np.array([
    "aa",
    "ap",
    "ba",
    "bi",
    "bu",
    "c",
    "d",
    "e",
    "h",
    "io",
    "ir",
    "me",
    "ma",
    "po",
    "ph",
    "pi",
    "ri",
    "sb",
    "se",
    "tw",
    "te",
    "th",
    "ti",
    "wd",
    "wi",
    "wr",
    "ww",
    "y"])
algorithms = ['MV', 'RF', 'wMV', 'I']
mappings = ['vol', 'inv']
divs = [20, 40, 60]
measurements = ['acc', 'precision', 'recall', 'fscore', 'specificity', 'auc']
displacements = [1, 5]
clfs = [3, 5]
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
ANY = 0


class Etl:

    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def read_cube(self):
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
                        assert len(values) == (
                                n_algorithms_independent + n_algorithms_mapping_dep * n_mappings + n_algorithms_mapping_div_dep * n_mappings * n_divs) * n_measurements
                        for (algorithm_variation_index, algorithm_values) in enumerate(self.chunker(values, n_measurements)):
                            print(f"filename nr: {file_index}, displacements: {displacements[n]}, clfs: {clfs[o]}, values: {algorithm_values}")
                            if file_index == 27:
                                self.print_conf_matrix(self.calculate_conf_matrix(float(algorithm_values[0]), float(algorithm_values[2]), float(algorithm_values[3]), 100))
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

    def calculate_conf_matrix(self, acc, precision, recall, size):
        tp = size * (1 - acc) / (1 / precision + 1 / recall - 2)
        return [
            [
                tp,
                size * (1 - acc) * (1 - precision) / (precision * (1 / precision + 1 / recall - 2))
            ],
            [
                size * (1 - acc) * (1 / recall - 1) / (1 / precision + 1 / recall - 2),
                size * (acc - (1 - acc) / (1 / precision + 1 / recall - 2))
            ]
        ]

    def calculate_conf_matrix_binary(self, acc, mcc, size, positive):
        if mcc == 0 and round(acc * size) == positive: return self.one_row_matrix(size, positive)
        ap = pow(mcc, 2) * size * positive - pow(mcc, 2) * pow(positive, 2) + pow(size, 2) / 4 - size * positive + pow(positive, 2)
        bp = acc * pow(size, 3) / 2 - acc * pow(size, 2) * positive - pow(mcc, 2) * pow(size, 2) * positive + pow(mcc, 2) * size * pow(positive, 2) - pow(size, 3) / 2 + 3 * pow(
            size, 2) * positive / 2 - size * pow(positive, 2)
        cp = pow(acc, 2) * pow(size, 4) / 4 - acc * pow(size, 4) / 2 + acc * pow(size, 3) * positive / 2 + pow(size, 4) / 4 - pow(size, 3) * positive / 2 + pow(size, 2) * pow(
            positive, 2) / 4
        print(f'ap: {ap}, bp: {bp}, cp: {cp}')
        deltasq = sqrt(pow(bp, 2) - 4 * ap * cp)
        x1, x2 = (-bp + deltasq) / (2 * ap), (-bp - deltasq) / (2 * ap)
        tp1, tp2 = (size * (acc - 1) + positive + x1) / 2, (size * (acc - 1) + positive + x2) / 2
        fp1, fn1, tn1 = round(x1 - tp1), round(positive - tp1), round(acc * size - tp1)
        fp2, fn2, tn2 = round(x2 - tp2), round(positive - tp2), round(acc * size - tp2)
        if (abs(self.mcc(tp1, fp1, fn1, tn1) - mcc) > abs(self.mcc(tp2, fp2, fn2, tn2) - mcc)) and (x2 > 0):
            tp, x = tp2, x2
        else:
            tp, x = tp1, x1
        return [
            [
                round(tp),
                round(x - tp)
            ],
            [
                round(positive - tp),
                round(acc * size - tp)
            ]
        ]

    def print_conf_matrix(self, matrix, verbose = False):
        t = Texttable()
        t.set_deco(0b1101)
        if verbose:
            t.add_rows([['', '', 'Actual class', ''], ['', '', 'P', 'N'], ['Predicted class', 'P', matrix[0][0], matrix[0][1]], ['', 'N', matrix[1][0], matrix[1][1]]])
        else:
            t.add_rows([[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]])
        print(t.draw())

    def one_row_matrix(self, size, positive):
        return [
            [positive, size - positive],
            [0, 0]
        ]

    def mcc(self, tp, fp, fn, tn):
        denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            return 0
        else:
            return (tp * tn - fp * fn) / denom
