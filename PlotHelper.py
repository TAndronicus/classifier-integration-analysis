import numpy as np


def get_plot_limits(lists: [], margin: float = .5, is_log: bool = False):
    """Returns limits for plot given offset

    :param lists: list of lists to be plotted
    :param margin: percentage of free space around plot
    :param is_log: is the plot logarithmic
    :return:
    """
    mins, maxes = np.zeros(len(lists), dtype = float), np.zeros(len(lists), dtype = float)
    for i in range(len(lists)):
        mins[i], maxes[i] = np.min(lists[i]), np.max(lists[i])
    min_min, max_max = np.min(mins), np.max(maxes)
    if is_log:
        plot_min, plot_max = min_min - np.power(10, margin) * (max_max - min_min), max_max + np.power(10, margin) * (max_max - min_min)
    else:
        plot_min, plot_max = min_min - margin * (max_max - min_min), max_max + margin * (max_max - min_min)
    return plot_min, plot_max
