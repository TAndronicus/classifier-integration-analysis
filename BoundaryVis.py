import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

base_models = {
    'base1': [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1]
    ],
    'base2': [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1]
    ],
    'base3': [
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1]
    ]
}
integrated_model = [
    [0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1]
]

# config
region_colors = {
    0: 'lightgray',
    1: 'gray'
}
boundary_color = 'b'
include_boundaries = True
incluse_mv = True


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


def calculate_boundaries(regions):
    return np.linspace(0, 1, len(regions[0]) + 1), np.linspace(0, 1, len(regions) + 1)


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


def show(x_lim=[0, 1], y_lim=[0, 1]):
    axes = plt.gca()
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    plt.show()


def plot_model(model):
    plot_regions(model)
    if include_boundaries: plot_boundaries(model)
    show()


# script
for model in base_models.values():
    plot_model(model)
if incluse_mv: plot_model(calculate_mv(base_models))
plot_model(integrated_model)
