import FileHelper
import PlotHelper
import os
import matplotlib.pyplot as plt
import numpy as np


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
        xlab = 'Liczba wyselekcjonowanych klasyfikatorów'
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
        ax = plt.subplot(1, 2, 1)
        ax.scatter(x, y)
        ax.scatter(x, z)
        ax.legend(['Majority voting', 'Integrated classifier'])
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
        ax.legend(['Majority voting', 'Integrated classifier'])
        ax.set_xlabel(xlab)
        ax.set_ylabel('Jakość klasyfikatora (MCC)')
        y_min, y_max = PlotHelper.get_plot_limits([y, z])
        ax.set_ylim(y_min, y_max)
        plt.show()


plot_dependence(filename = "biodeg.scsv", space_parts = 3, n_class = 9, n_best = 2, n_best_diff = 1,
                    i_meth = 0, bagging = 0, dependency = "n_best")
