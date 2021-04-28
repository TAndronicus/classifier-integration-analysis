import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import os

base_models = {
    'base1': [
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ],
    'base2': [
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ],
    'base3': [
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1]
    ]
}
integrated_model = [
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1]
]

# config
region_colors = {
    0: 'r',
    1: 'b'
}
boundary_color = 'b'
include_boundaries = True
incluse_mv = True
save_to_file = True
figures_dir = 'fig'
custom_regions = True


# defs
def plot_regions(regions):
    x_boundaries, y_boundaries = calculate_boundaries(regions)
    for x_ind in range(0, len(regions[0])):
        for y_ind in range(0, len(regions)):
            plt.fill(
                [x_boundaries[x_ind], x_boundaries[x_ind], x_boundaries[x_ind + 1], x_boundaries[x_ind + 1]],
                [y_boundaries[y_ind], y_boundaries[y_ind + 1], y_boundaries[y_ind + 1], y_boundaries[y_ind]],
                color=region_colors.get(get_cell(regions, x_ind, y_ind))
            )


def get_cell(regions, x_ind, y_ind):
    return regions[-y_ind - 1][x_ind]


def custom_boundaries():
    return [0, .2, .3, .4, .6, .7, .9, 1], [0, .2, .4, .6, .8, 1]


def calculate_boundaries(regions):
    if custom_regions: return custom_boundaries()
    else: return np.linspace(0, 1, len(regions[0]) + 1), np.linspace(0, 1, len(regions) + 1)


def plot_boundaries(regions):
    plot_horizontal_boundaries(regions)
    plot_vertical_boundaries(regions)


def plot_horizontal_boundaries(regions):
    x_boundaries, y_boundaries = calculate_boundaries(regions)
    for x_ind in range(0, len(regions[0])):
        for y_ind in range(0, len(regions) - 1):
            if get_cell(regions, x_ind, y_ind) != get_cell(regions, x_ind, y_ind + 1):
                plt.plot(
                    [x_boundaries[x_ind], x_boundaries[x_ind + 1]],
                    [y_boundaries[y_ind + 1], y_boundaries[y_ind + 1]],
                    color=boundary_color,
                )


def plot_vertical_boundaries(regions):
    x_boundaries, y_boundaries = calculate_boundaries(regions)
    for x_ind in range(0, len(regions[0]) - 1):
        for y_ind in range(0, len(regions)):
            if get_cell(regions, x_ind, y_ind) != get_cell(regions, x_ind + 1, y_ind):
                plt.plot(
                    [x_boundaries[x_ind + 1], x_boundaries[x_ind + 1]],
                    [y_boundaries[y_ind], y_boundaries[y_ind + 1]],
                    color=boundary_color,
                )


def calculate_mv(bases):
    return (reduce(
        lambda m1, m2: m1 + m2,
        map(
            lambda ll: np.matrix(ll),
            bases.values()
        )
    ) / len(bases)).round().tolist()


def create_dir_if_not_exists():
    if not(os.path.exists(figures_dir)):
        os.makedirs(figures_dir)


def show(name, x_lim=[0, 1], y_lim=[0, 1]):
    axes = plt.gca()
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    if save_to_file:
        plt.savefig(figures_dir + '/' + name + '.png')
    else:
        plt.show()
    plt.clf()


def plot_model(model, name):
    plot_regions(model)
    if include_boundaries: plot_boundaries(model)
    show(name)


# script
if save_to_file: create_dir_if_not_exists()
for name, model in base_models.items():
    plot_model(model, name)
if incluse_mv: plot_model(calculate_mv(base_models), 'mv')
plot_model(integrated_model, 'integrated')
