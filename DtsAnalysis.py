import os
from DtsRes import DtsRes
from nonparametric_tests import friedman_test, bonferroni_dunn_test, holm_test

seriex = ['simple', 'pre', 'post-cv', 'post-training']
filenames = ["bi", "bu", "c", "d", "h", "i", "m", "p", "se", "t", "wd", "wi"]
n_clfs = [3, 5, 7, 9]
alphas = [".3", "1.0"]
dims = ["clf", "alpha", "series"]


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
    if not(dim in dims):
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


objs = get_dependent_on(dims[2], 7, 60, seriex[0])
objs = map_dtrex(objs, "mcc")
iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test(objs)
print(rankings_cmp)
rankings = create_rank_dict(rankings_cmp)
comparisonsH, z, pH, adj_p = holm_test(rankings, str(len(rankings) - 1))
print(pH)
# for o in objs:
#     for oin in o:
#         print(str(oin))
#     print("\n")
