import os

import matplotlib.pyplot as plt

import FileHelper
import PlotHelper

filenames = ['biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx',
             'data_banknote_authentication.csv', 'haberman.dat',
             'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv',
             'seismic_bumps.dat', 'twonorm.dat', 'wdbc.dat',
             'wisconsin.dat']

spaces = [3, 4, 5, 6, 7, 8, 9, 10]
clfs = [3, 4, 5, 6, 7, 8, 9]
method_dict_short = {
    0 : "A",
    1 : "M"
}
method_dict_long = {
    0 : "Weighted average",
    1 : "Median"
}

def read_in_objects():
    name_pattern = "n_{}_b_{}_i_{}.xls"
    n_class_range = list(range(3, 10, 2))
    i_meth_range = [0, 1]
    bagging_range = [0, 1]
    result_objects = []
    for n_class in n_class_range:
        for bagging in bagging_range:
            for i_meth in i_meth_range:
                res_filename = name_pattern.format(n_class, bagging, i_meth)
                absolute_path = os.path.join(os.path.dirname(__file__), "results/" + res_filename)
                file_objects = FileHelper.read_objects_from_file(absolute_path, n_class, bagging, i_meth)
                result_objects.append(file_objects)
    result_objects = [item for sublist in result_objects for item in sublist]
    return result_objects


def get_dependent_on_space_parts(filename, n_class, n_best, i_meth, bagging):
    result_objects = read_in_objects()
    related = []
    for obj in result_objects:
        if obj.filename == filename \
                and obj.n_class == n_class \
                and obj.n_best == n_best \
                and obj.i_meth == i_meth \
                and obj.bagging == bagging:
            related.append(obj)
    return related


def get_dependent_on_n_classif_const_n_best(filename, space_parts, n_best, i_meth, bagging):
    result_objects = read_in_objects()
    related = []
    for obj in result_objects:
        if obj.filename == filename \
                and obj.space_parts == space_parts \
                and obj.n_best == n_best \
                and obj.i_meth == i_meth \
                and obj.bagging == bagging:
            related.append(obj)
    return related


def get_dependent_on_n_classif_non_const_n_best(filename, space_parts, n_best_diff, i_meth, bagging):
    result_objects = read_in_objects()
    related = []
    for obj in result_objects:
        if obj.filename == filename \
                and obj.space_parts == space_parts \
                and obj.n_class - obj.n_best == n_best_diff \
                and obj.i_meth == i_meth \
                and obj.bagging == bagging:
            related.append(obj)
    return related


def get_dependent_on_n_best(filename, space_parts, n_class, i_meth, bagging):
    result_objects = read_in_objects()
    related = []
    for obj in result_objects:
        if obj.filename == filename \
                and obj.space_parts == space_parts \
                and obj.n_class == n_class \
                and obj.i_meth == i_meth \
                and obj.bagging == bagging:
            related.append(obj)
    return related


def get_dependent_on_filename(space_parts, n_class, n_best, i_meth, bagging):
    result_objects = read_in_objects()
    related = []
    for obj in result_objects:
        if obj.space_parts == space_parts \
                and obj.n_class == n_class \
                and obj.n_best == n_best \
                and obj.i_meth == i_meth \
                and obj.bagging == bagging:
            related.append(obj)
    return related


def plot_dependence(filename: str = "biodeg.scsv",
                    space_parts: int = 3,
                    n_class: int = 3,
                    n_best: int = 2,
                    n_best_diff: int = 1,
                    i_meth: int = 0,
                    bagging: int = 0,
                    dependency: str = "space_parts"):
    if dependency == "space_parts":
        related = get_dependent_on_space_parts(filename, n_class, n_best, i_meth, bagging)
        attr = "space_parts"
        xlab = 'Liczba podprzestrzeni'
    elif dependency == "n_class_const_n_best":
        related = get_dependent_on_n_classif_const_n_best(filename, space_parts, n_best, i_meth, bagging)
        attr = "n_class"
        xlab = 'Liczba bazowych klasyfikatorów'
    elif dependency == "n_class_non_const_n_best":
        related = get_dependent_on_n_classif_non_const_n_best(filename, space_parts, n_best_diff, i_meth, bagging)
        attr = "n_class"
        xlab = 'Liczba bazowych klasyfikatorów'
    elif dependency == "n_best":
        related = get_dependent_on_n_best(filename, space_parts, n_class, i_meth, bagging)
        attr = "n_best"
        xlab = 'Liczba klasyfikatorów po selekcji'
    else:
        related = get_dependent_on_space_parts(filename, n_class, n_best, i_meth, bagging)
        attr = "space_parts"
        xlab = 'Liczba podprzestrzeni'
    if len(related) != 0:
        x, y, z = [], [], []
        for obj in related:
            x.append(getattr(obj, attr))
            y.append(obj.mv_score)
            z.append(obj.i_score)
        ax = plt.subplot(1, 2, 1, )
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['MV', 'IC'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (ACC)')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        y, z = [], []
        for obj in related:
            y.append(obj.mv_mcc)
            z.append(obj.i_mcc)
        ax = plt.subplot(1, 2, 2)
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['MV', 'IC'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (MCC)')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        plt.show()


def plot_bagging_difference(
        filename: str = "biodeg.scsv",
        space_parts: int = 3,
        n_class: int = 3,
        n_best: int = 2,
        n_best_diff: int = 1,
        i_meth: int = 0,
        dependency: str = "space_parts"):
    if dependency == "space_parts":
        attr = 'space_parts'
        xlab = 'Liczba podprzestrzeni'
        non_bagging = get_dependent_on_space_parts(filename, n_class, n_best, i_meth, 0)
        bagging = get_dependent_on_space_parts(filename, n_class, n_best, i_meth, 1)
    elif dependency == "n_class_const_n_best":
        attr = 'n_class'
        xlab = 'Liczba klasyfikatorów bazowych'
        non_bagging = get_dependent_on_n_classif_const_n_best(filename, space_parts, n_best, i_meth, 0)
        bagging = get_dependent_on_n_classif_const_n_best(filename, space_parts, n_best, i_meth, 1)
    elif dependency == "n_class_non_const_n_best":
        attr = 'n_class'
        xlab = 'Liczba klasyfikatorów bazowych'
        non_bagging = get_dependent_on_n_classif_non_const_n_best(filename, space_parts, n_best_diff, i_meth, 0)
        bagging = get_dependent_on_n_classif_non_const_n_best(filename, space_parts, n_best_diff, i_meth, 1)
    elif dependency == "n_best":
        attr = "n_best"
        xlab = 'Liczba klasyfikatorów po selekcji'
        non_bagging = get_dependent_on_n_best(filename, space_parts, n_class, i_meth, 0)
        bagging = get_dependent_on_n_best(filename, space_parts, n_class, i_meth, 1)
    else:
        attr = 'space_parts'
        xlab = 'Liczba podprzestrzeni'
        non_bagging = get_dependent_on_space_parts(filename, n_class, n_best, i_meth, 0)
        bagging = get_dependent_on_space_parts(filename, n_class, n_best, i_meth, 1)
    if len(non_bagging) == len(bagging) != 0:
        x, y, z = [], [], []
        for i in range(len(non_bagging)):
            if getattr(non_bagging[i], attr) != getattr(bagging[i], attr):
                print('dafuq')
            x.append(getattr(non_bagging[i], attr))
            y.append(non_bagging[i].i_score)
            z.append(bagging[i].i_score)
        ax = plt.subplot(1, 2, 1)
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['No bagging', 'Bagging'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (ACC)')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        y, z = [], []
        for i in range(len(non_bagging)):
            y.append(non_bagging[i].i_mcc)
            z.append(bagging[i].i_mcc)
        ax = plt.subplot(1, 2, 2)
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['No bagging', 'Bagging'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (MCC)')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        plt.show()


def plot_method_difference(
        filename: str = "biodeg.scsv",
        space_parts: int = 3,
        n_class: int = 3,
        n_best: int = 2,
        n_best_diff: int = 1,
        bagging: int = 0,
        dependency: str = "space_parts"):
    if dependency == "space_parts":
        attr = 'space_parts'
        xlab = 'Liczba podprzestrzeni'
        i_mean = get_dependent_on_space_parts(filename, n_class, n_best, 0, bagging)
        i_median = get_dependent_on_space_parts(filename, n_class, n_best, 1, bagging)
    elif dependency == "n_class_const_n_best":
        attr = 'n_class'
        xlab = 'Liczba klasyfikatorów bazowych'
        i_mean = get_dependent_on_n_classif_const_n_best(filename, space_parts, n_best, 0, bagging)
        i_median = get_dependent_on_n_classif_const_n_best(filename, space_parts, n_best, 1, bagging)
    elif dependency == "n_class_non_const_n_best":
        attr = 'n_class'
        xlab = 'Liczba klasyfikatorów bazowych'
        i_mean = get_dependent_on_n_classif_non_const_n_best(filename, space_parts, n_best_diff, 0, bagging)
        i_median = get_dependent_on_n_classif_non_const_n_best(filename, space_parts, n_best_diff, 1, bagging)
    elif dependency == "n_best":
        attr = "n_best"
        xlab = 'Liczba klasyfikatorów po selekcji'
        i_mean = get_dependent_on_n_best(filename, space_parts, n_class, 0, bagging)
        i_median = get_dependent_on_n_best(filename, space_parts, n_class, 1, bagging)
    else:
        attr = 'space_parts'
        xlab = 'Liczba podprzestrzeni'
        i_mean = get_dependent_on_space_parts(filename, n_class, n_best, 0, bagging)
        i_median = get_dependent_on_space_parts(filename, n_class, n_best, 1, bagging)
    if len(i_mean) == len(i_median) != 0:
        x, y, z = [], [], []
        for i in range(len(i_mean)):
            if getattr(i_mean[i], attr) != getattr(i_median[i], attr):
                print('dafuq')
            x.append(getattr(i_mean[i], attr))
            y.append(i_mean[i].i_score)
            z.append(i_median[i].i_score)
        ax = plt.subplot(1, 2, 1)
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['Średnia', 'Mediana'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (ACC)')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        y, z = [], []
        for i in range(len(i_mean)):
            y.append(i_mean[i].i_mcc)
            z.append(i_median[i].i_mcc)
        ax = plt.subplot(1, 2, 2)
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['Średnia', 'Mediana'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (MCC)')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        plt.show()


def print_tables():
    # global space
    for space in range(3, 11):
        dep_wa = get_dependent_on_filename(space, 9, 8, 0, 0)
        dep_m = get_dependent_on_filename(space, 9, 8, 1, 0)
        with open('stats' + str(space), 'w') as file:
            file.write(
                ',ACC,,,MCC,,\n,$\\Psi_{\mathrm{MV}}$,$\\Psi_{\mathrm{WA}}$,$\\Psi_{\mathrm{M}}$,$\\Psi_{\mathrm{MV}}$,$\\Psi_{\mathrm{WA}}$,$\\Psi_{\mathrm{M}}$\n')
            for i in range(len(dep_wa)):
                file.write(str(dep_m[i].filename).split('.')[0] + ',' + str(round_to_3(dep_m[i].mv_score)) + ',' + str(round_to_3(dep_wa[i].i_score)) + ',' + str(
                    round_to_3(dep_m[i].i_score)) + ',' + str(round_to_3(dep_m[i].mv_mcc)) + ',' + str(round_to_3(dep_wa[i].i_mcc)) + ',' + str(round_to_3(dep_m[i].i_mcc)) + '\n')


# for o in obj:


def get_obj(n_class, n_best, n_space):
    obj = read_in_objects()
    res = []
    for o in obj:
        if o.n_class == n_class and o.n_best == n_best and o.space_parts == n_space and o.bagging == 0:
            res.append(o)
    return res


def get_one(n_class, n_best, n_space, filename, meth):
    obj = read_in_objects()
    for o in obj:
        if o.n_class == n_class and o.n_best == n_best and o.space_parts == n_space and o.bagging == 0 and o.filename == filename and o.i_meth == meth:
            return o


"""
Sprawdza różnicę między populacjami dla zmieniającej się gęstości podziałów
"""
def check_space():
    for clf in clfs:
        for best in range(2, clf):
            forwards = 0
            backwards = 0
            comp = 0
            worse = 0
            for index in range(0, len(spaces) - 1):
                o1 = get_obj(clf, best, spaces[index])
                o2 = get_obj(clf, best, spaces[index + 1])
                if len(o1) != len(o2):
                    print("Różne dł")
                for i in range(0, len(o1)):
                    if o1[i].i_score < o2[i].i_score:
                        forwards += 1
                    if o1[i].i_score > o2[i].i_score:
                        backwards += 1
                    if o1[i].i_score > o1[i].mv_score:
                        comp += 1
                    if o1[i].i_score < o1[i].mv_score:
                        worse += 1
            lo = get_obj(clf, best, spaces[-1])
            for o in lo:
                if o.i_score > o.mv_score:
                    comp += 1
                if o.i_score < o.mv_score:
                    worse += 1
            if forwards + backwards == 0:
                continue
            print("clfs: " + str(clf) + ", best: " + str(best) + " ::: forwards : " + str(forwards) + ", backwards : " + str(backwards) + ", better : " + str(
                comp) + ", worse : " + str(worse))


"""
Sprawdza różnicę między populacjami dla zmieniającej się gęstości podziałów
"""
def check_best():
    global clf, space, o
    for clf in clfs:
        for space in spaces:
            forwards = 0
            backwards = 0
            comp = 0
            worse = 0
            for index in range(2, clf - 1):
                o1 = get_obj(clf, index, space)
                o2 = get_obj(clf, index + 1, space)
                if len(o1) != len(o2):
                    print("Różne dł")
                for i in range(0, len(o1)):
                    if o1[i].i_score < o2[i].i_score:
                        forwards += 1
                    if o1[i].i_score > o2[i].i_score:
                        backwards += 1
                    if o1[i].i_score > o1[i].mv_score:
                        comp += 1
                    if o1[i].i_score < o1[i].mv_score:
                        worse += 1
            ol = get_obj(clf, clf - 1, space)
            for o in ol:
                if o.i_score > o.mv_score:
                    comp += 1
                if o.i_score < o.mv_score:
                    worse += 1
            if forwards + backwards == 0:
                continue
            print(
                "clfs: " + str(clf) + ", space: " + str(space) + " ::: forwards : " + str(forwards) + ", backwards : " + str(backwards) + ", better : " + str(
                    comp) + ", worse : " + str(worse))


def round_to_3(a: float):
    return int(a * 1000) / 1000


"""
Prints spaces for which score is different by at least 0,0005
"""
def show_diff_space(clf1, clf2, best1, best2):
    for space in spaces:
        for filename in filenames:
            for method in range(0, 2):
                print(filename)
                o1 = get_one(clf1, best1, space, filename, method)
                o2 = get_one(clf2, best2, space, filename, method)
                if round_to_3(o1.i_score) != round_to_3(o2.i_score):
                    print("space: " + str(space) + ", ds: " + o1.filename)

