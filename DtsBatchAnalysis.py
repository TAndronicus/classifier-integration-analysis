import os

from DtsBatchRes import DtsBatchRes
from MathUtils import round_to_str
from nonparametric_tests import friedman_test, bonferroni_dunn_test

seriex = ["pre-tr"]
filenames = ['bio', 'bup', 'cry', 'dba', 'hab', 'ion', 'met', 'pop', 'sei', 'wdb', 'wis']
references = ['mv', 'rf']
n_clfs = [3, 5, 7, 9]
alphas = ["0.0", "0.3", "0.7", "1.0"]
betas1 = ["0.5"]
betas2 = ["0.0"]
gammas1 = ["20.0"]
gammas2 = ["5.0"]
dims = ["clf", "alpha", "series"]


def read(n_clf, alpha, series):
    name_pattern = "dts/" + series + "/{}_{}_{}_{}_{}_{}"
    res_filename = name_pattern.format(n_clf, alpha, betas1[0], betas2[0], gammas1[0], gammas2[0])
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    objects = []
    with(open(absolute_path)) as file:
        counter = 0
        for line in file.readlines():
            values = line.split(',')
            if len(values) < 6: continue
            obj = DtsBatchRes(float(values[0]), float(values[1]),  # mv
                              float(values[2]), float(values[3]),  # rf
                              float(values[4]), float(values[5]),  # i
                              n_clf, alpha, filenames[counter])
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


def print_results(file_to_write = None):
    def_series = seriex[0]
    for n_clf in n_clfs:
        for meas in ['acc', 'mcc']:
            custom_print('\nn_clf: ' + str(n_clf) + ', meas: ' + meas + '\n', file_to_write)

            for filename_index in filenames:
                custom_print(',' + filename_index, file_to_write)
            custom_print(',rank\n', file_to_write)

            objs = []
            for alpha in alphas:
                objs.append(read(n_clf, alpha, def_series))
            values = map_dtrex(objs, meas)
            iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)

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
                    custom_print(round_to_str(values[counter][filename_index], 3) + ',', file_to_write)
                custom_print(round_to_str(rankings_cmp[counter], 2) + '\n', file_to_write)
                counter = counter + 1

            ## post-hoc
            rankings = create_rank_dict(rankings_cmp)
            comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, '0')
            pH = [x for _, x in sorted(zip(comparisonsH, pH))]
            custom_print('p-values: ' + str(pH) + '\n', file_to_write)


with open('reports/1-batch.csv', 'w') as f:
    print_results(f)
