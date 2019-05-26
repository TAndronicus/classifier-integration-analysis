import os
from DtsRes import DtsRes
from nonparametric_tests import friedman_test, bonferroni_dunn_test, holm_test
from MathUtils import round_to_str

seriex = ["sim", "pre", "post-cv", "post-tr"]
effective_seriex = ["pre", "post-cv", "post-tr"]
filenames = ["bi", "bu", "c", "d", "h", "i", "m", "p", "se", "t", "wd", "wi"]
n_clfs = [3, 5, 7, 9]
alphas = ["1.0", "0.3"]
dims = ["clf", "alpha", "series", "effective_series"]


def read(n_clf, alpha, series):
    name_pattern = "dt/" + series + "/{}_{}"
    res_filename = name_pattern.format(n_clf, alpha)
    absolute_path = os.path.join(os.path.dirname(__file__), res_filename)
    objects = []
    with(open(absolute_path)) as file:
        counter = 0
        for line in file.readlines():
            values = line.split(",")
            obj = DtsRes(float(values[0]), float(values[1]), float(values[2]), float(values[3]), n_clf, alpha, filenames[counter])
            objects.append(obj)
            counter += 1
    return objects


def get_dependent_on(dim, n_clf, alpha, series):
    if not (dim in dims):
        raise Exception("Wrong dim")
    if dim == dims[0]:
        objs = []
        for nc in n_clfs:
            objs.append(read(nc, alpha, series))
        return objs
    if dim == dims[1]:
        objs = []
        for a in alphas:
            objs.append(read(n_clf, a, series))
        return objs
    if dim == dims[2]:
        objs = []
        for ns in seriex:
            objs.append(read(n_clf, alpha, ns))
        return objs
    if dim == dims[3]:
        objs = []
        for ns in effective_seriex:
            objs.append(read(n_clf, alpha, ns))
        return objs


def get_average_mv(objects, attr):
    res_out = []
    length = len(objects)
    for i in range(len(objects[0])):
        value = 0
        for j in range(length):
            value += getattr(objects[j][i], "mv_" + attr)
        res_out.append(value / length)
    return res_out


def map_dtrex(objects, attr):
    res_out = []
    for obj_out in objects:
        res_in = []
        for obj_in in obj_out:
            res_in.append(getattr(obj_in, "i_" + attr))
        res_out.append(res_in)
    res_out.append(get_average_mv(objects, attr))
    return res_out


def create_rank_dict(rankings):
    dict = {}
    for i in range(len(rankings)):
        dict[str(i)] = rankings[i]
    return dict


def find_first_by_filename(objects, filename):
    for object in objects:
        if object.filename == filename:
            return object
    raise Exception("Filename not found: " + filename)


def print_stats_n_clf():
    dependent_dim = dims[0]
    for series in seriex:
        print(series)
        for meas in ["acc", "mcc"]:
            print(meas)
            objs = get_dependent_on(dependent_dim, n_clfs[0], alphas[1], series)
            objs = map_dtrex(objs, meas)
            iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(objs)
            print("ranks: " + str(rankings_cmp))
            rankings = create_rank_dict(rankings_cmp)
            comparisonsH, z, pH, adj_p = holm_test(rankings, str(len(rankings) - 1))
            pH = [x for _, x in sorted(zip(comparisonsH, pH))]
            print("p-values: " + str(pH))


def print_stats_series(file = None):
    dependent_dim = dims[3]
    for alpha in alphas:
        for n_clf in n_clfs:
            for meas in ["acc", "mcc"]:
                custom_print("\nalpha: " + alpha + ", meas: " + meas + ", n_clf: " + str(n_clf) + "\n", file)
                objs = get_dependent_on(dependent_dim, n_clf, alpha, effective_seriex[0])
                objs = map_dtrex(objs, meas)
                iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(objs)
                custom_print("ranks: " + str(rankings_cmp) + "\n", file)
                rankings = create_rank_dict(rankings_cmp)
                comparisonsH, z, pH, adj_p = holm_test(rankings, str(len(rankings) - 1))
                pH = [x for _, x in sorted(zip(comparisonsH, pH))]
                custom_print("p-values: " + str(pH) + "\n", file)


def custom_print(text, file = None):
    if file is None:
        print(text, end = "")
    else:
        file.write(text)


def get_sums_by_filenames():
    res = {}
    for filename in filenames:
        res[filename] = 0
    return res


def print_results(file_to_write):
    dependent_dim = dims[3]
    for alpha in alphas:
        for meas in ["acc", "mcc"]:
            for n_clf in n_clfs:
                custom_print("\nalpha: " + alpha + ", meas: " + meas + ", n_clf: " + str(n_clf) + "\n", file_to_write)
                custom_print("series", file_to_write)

                for filename in filenames:
                    custom_print("," + filename, file_to_write)
                custom_print(",rank\n", file_to_write)

                objs_all_series = get_dependent_on(dependent_dim, n_clf, alpha, effective_seriex[0])
                values = map_dtrex(objs_all_series, meas)
                iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(values)

                counter = 0
                sum_by_filename = get_sums_by_filenames()
                for series in effective_seriex:
                    custom_print(series + ",", file_to_write)
                    objs = read(n_clf, alpha, series)
                    for filename in filenames:
                        obj = find_first_by_filename(objs, filename)
                        custom_print(round_to_str(getattr(obj, "i_" + meas), 3) + ",", file_to_write)
                        sum_by_filename[filename] = sum_by_filename[filename] + getattr(obj, "mv_" + meas)
                    custom_print(round_to_str(rankings_cmp[counter], 2) + "\n", file_to_write)
                    counter = counter + 1

                custom_print("mv,", file_to_write)
                for filename in filenames:
                    custom_print(round_to_str(sum_by_filename[filename] / len(effective_seriex), 3) + ",", file_to_write)
                custom_print(round_to_str(rankings_cmp[counter], 2) + "\n", file_to_write)
                custom_print("Friedman p-val: " + str(p_value) + "\n", file_to_write)


with open("1-res.csv", "w") as file:
    print_results(file)
with open("1-stats.csv", "w") as f:
    print_stats_series(f)
