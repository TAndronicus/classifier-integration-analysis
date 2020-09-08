import os

import numpy as np

from DtdDisplacementAnalysisNp import n_filenames, n_algorithms_independent, n_algorithms_mapping_dep, n_algorithms_mapping_div_dep, n_mappings, n_divs, n_measurements, \
    n_displacements, clfs, displacements, divs, num_features, has_header

header = 'MV(acc),MV(precision),MV(recall),MV(fscore),MV(specificity),MV(auc),RF(acc),RF(precision),RF(recall),RF(fscore),RF(specificity),RF(auc),wMV_vol(acc),wMV_vol(precision),wMV_vol(recall),wMV_vol(fscore),wMV_vol(specificity),wMV_vol(auc),wMV_inv(acc),wMV_inv(precision),wMV_inv(recall),wMV_inv(fscore),wMV_inv(specificity),wMV_inv(auc),I_vol^20(acc),I_vol^20(precision),I_vol^20(recall),I_vol^20(fscore),I_vol^20(specificity),I_vol^20(auc),I_inv^20(acc),I_inv^20(precision),I_inv^20(recall),I_inv^20(fscore),I_inv^20(specificity),I_inv^20(auc),I_vol^40(acc),I_vol^40(precision),I_vol^40(recall),I_vol^40(fscore),I_vol^40(specificity),I_vol^40(auc),I_inv^40(acc),I_inv^40(precision),I_inv^40(recall),I_inv^40(fscore),I_inv^40(specificity),I_inv^40(auc),I_vol^60(acc),I_vol^60(precision),I_vol^60(recall),I_vol^60(fscore),I_vol^60(specificity),I_vol^60(auc),I_inv^60(acc),I_inv^60(precision),I_inv^60(recall),I_inv^60(fscore),I_inv^60(specificity),I_inv^60(auc)'

ref_cols = n_algorithms_independent * n_measurements
for (o, clf) in enumerate(clfs):
    res = np.zeros(
        (n_filenames, (n_algorithms_independent + n_algorithms_mapping_dep * n_mappings + n_algorithms_mapping_div_dep * n_mappings * n_divs) * n_measurements, n_displacements))
    for (n, displacement) in enumerate(displacements):
        name_pattern = 'dtd-displacement/{}_{}_[' + '_'.join([str(el) for el in divs]) + ']_{}'
        res_filename = name_pattern.format(clf, num_features, displacement)
        absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
        with(open(absolute_path)) as file:
            for (line_num, line) in enumerate(file.readlines()):
                if has_header and line_num == 0: continue
                file_index = line_num - [0, 1][has_header]
                values = line.split(',')
                res[file_index, :, n] = values
    res_agg = np.average(res, axis = 2)
    for (n, displacement) in enumerate(displacements):
        name_pattern = 'dtd-displacement/{}_{}_[' + '_'.join([str(el) for el in divs]) + ']_{}'
        res_filename = name_pattern.format(clf, num_features, displacement)
        absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
        with(open(absolute_path, 'w')) as file:
            file.write(header + '\n')
            for line_num in range(n_filenames):
                file.write(','.join(['%.4f' % num for num in res_agg[line_num, :ref_cols]]) + ',' + ','.join(['%.4f' % num for num in res[line_num, ref_cols:, n]]) + '\n')
