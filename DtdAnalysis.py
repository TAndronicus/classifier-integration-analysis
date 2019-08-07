import os
from DtdRes import DtdRes
from nonparametric_tests import friedman_test, bonferroni_dunn_test, holm_test
from LatexMappings import LatexMappings
from MathUtils import round_to_str

seriex = ['vol-shallow', 'vol-deep', 'inv-shallow', 'inv-deep']
# seriex = ['vol-shallow', 'inv-shallow']
# seriex = ['vol-deep', 'inv-deep']
filenames = ["bi", "bu", "c", "d", "h", "i", "m", "p", "se", "wd", "wi"]
n_clfs = [3, 5, 7, 9]
# n_feas = [2, 3]
n_feas = [2]
n_divs = [20, 40, 60]
dims = ["clf", "fea", "div", "series"]


def read(n_clf, n_fea, n_div, series):
    name_pattern = "dtd/" + series + "/{}_{}_{}"
    res_filename = name_pattern.format(n_clf, n_fea, n_div)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    objects = []
    with(open(absolute_path)) as file:
        counter = 0
        for line in file.readlines():
            values = line.split(",")
            obj = DtdRes(float(values[0]), float(values[1]), # mv
                         float(values[2]), float(values[3]), # rf
                         float(values[4]), float(values[5]), # i
                         n_clf, n_fea, n_div, filenames[counter])
            objects.append(obj)
            counter += 1
    return objects


def get_dependent_on(dim, n_clf, n_fea, n_div, series):
    if not (dim in dims):
        raise Exception("Wrong dim")
    if dim == dims[0]:
        objs = []
        for nc in n_clfs:
            objs.append(read(nc, n_fea, n_div, series))
        return objs
    if dim == dims[1]:
        objs = []
        for nf in n_feas:
            objs.append(read(n_clf, nf, n_div, series))
        return objs
    if dim == dims[2]:
        objs = []
        for nd in n_divs:
            objs.append(read(n_clf, n_fea, nd, series))
        return objs
    if dim == dims[3]:
        objs = []
        for ns in seriex:
            objs.append(read(n_clf, n_fea, n_div, ns))
        return objs


def get_average_reference(n_fea, attr, n_clf):
    objects = []
    for div in n_divs:
        for series in seriex:
            objects.append(read(n_clf, n_fea, div, series))
    mv_out = []
    rf_out = []
    length = len(objects)
    for i in range(len(objects[0])):
        mv_value, rf_value = 0, 0
        for j in range(length):
            mv_value += getattr(objects[j][i], "mv_" + attr)
            rf_value += getattr(objects[j][i], "rf_" + attr)
        mv_out.append(mv_value / length)
        rf_out.append(rf_value / length)
    return [mv_out, rf_out]


def map_dtrex(objects, n_fea, attr, n_clf):
    res_out = []
    for obj_out in objects:
        res_in = []
        for obj_in in obj_out:
            res_in.append(getattr(obj_in, "i_" + attr))
        res_out.append(res_in)
    for reference in get_average_reference(n_fea, attr, n_clf):
        res_out.append(reference)
    return res_out


def create_rank_dict(rankings):
    dict = {}
    for i in range(len(rankings)):
        dict[str(i)] = rankings[i]
    return dict


def print_stats_series(file = None):
    dependent_dim = dims[3]
    for n_fea in n_feas:
        for meas in ["acc", "mcc"]:
            for n_clf in n_clfs:
                custom_print("\nn_fea: " + str(n_fea) + ", n_clf: " + str(n_clf) + ", meas: " + meas + "\n", file)
                objs = get_dependent_on(dependent_dim, n_clf, n_fea, n_divs[0], seriex[0])
                objs = map_dtrex(objs, n_fea, meas, n_clf)
                iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(objs)
                rankings = create_rank_dict(rankings_cmp)
                custom_print("ranks: " + str(rankings_cmp) + "\n", file)
                comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, str(len(rankings) - 1))
                pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                custom_print("p-values: " + str(pH) + "\n", file)
                custom_print("friedman p-val: " + str(p_value) + "\n", file)


def custom_print(text, file = None):
    if file is None:
        print(text, end = "")
    else:
        file.write(text)


def find_first_by_filename(objects, filename):
    for object in objects:
        if object.filename == filename:
            return object
    raise Exception("Filename not found: " + filename)


def set_mapping(mapping = 0):
    global seriex
    if mapping == 0:
        seriex = ['vol-shallow', 'inv-shallow']
    elif mapping == 1:
        seriex = ['vol-deep', 'inv-deep']
    else:
        raise Exception("No such mapping")


def print_results(file_to_write = None, n_div = n_divs[0]):
    dependent_dim = dims[3]
    for n_fea in n_feas:
        for meas in ["acc", "mcc"]:
            for n_clf in n_clfs:
                custom_print("\nn_fea: " + str(n_fea) + ", meas: " + meas + ", n_clf: " + str(n_clf) + "\n", file_to_write)

                for filename in filenames:
                    custom_print("," + filename, file_to_write)
                custom_print(",rank\n", file_to_write)

                objs_all_series = get_dependent_on(dependent_dim, n_clf, n_fea, n_div, seriex[0])
                values = map_dtrex(objs_all_series, n_fea, meas, n_clf)
                iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)

                counter = 0
                for series in seriex:
                    custom_print(LatexMappings.dtd_series_names[series] + ",", file_to_write)
                    objs = read(n_clf, n_fea, n_div, series)
                    for filename in filenames:
                        obj = find_first_by_filename(objs, filename)
                        custom_print(round_to_str(getattr(obj, "i_" + meas), 3) + ",", file_to_write)
                    custom_print(round_to_str(rankings_cmp[counter], 2) + "\n", file_to_write)
                    counter = counter + 1

                custom_print(LatexMappings.dtd_series_names['mv-p'][mapping] + ",", file_to_write)
                for dataset in range(len(values[counter])):
                    custom_print(round_to_str(values[counter][dataset], 3) + ",", file_to_write)
                custom_print(round_to_str(rankings_cmp[counter], 2) + "\n", file_to_write)
                counter = counter + 1

                custom_print(LatexMappings.dtd_series_names['rf-p'][mapping] + ",", file_to_write)
                for dataset in range(len(values[counter])):
                    custom_print(round_to_str(values[counter][dataset], 3) + ",", file_to_write)
                custom_print(round_to_str(rankings_cmp[counter], 2) + "\n", file_to_write)

                ## post-hoc
                rankings = create_rank_dict(rankings_cmp)
                comparisonsH, z, pH, adj_p = bonferroni_dunn_test(rankings, str(len(rankings) - 1))
                pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                custom_print("p-values: " + str(pH) + "\n", file_to_write)


for n_div in n_divs:
    for mapping in [0, 1]:
        set_mapping(mapping)
        with open("2-res-" + str(n_div) + "-" + ["shallow", "deep"][mapping] + ".csv", "w") as f:
            print_results(f, n_div)
with open("2-stats.csv", "w") as f:
    print_stats_series(f)
