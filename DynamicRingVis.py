import itertools

import matplotlib.pyplot as plt

### Constants
colors = ['r', 'g', 'b']


### Functions
def draw_tree(coordinates: [], color = 'r'):
    for (p1, p2) in [(coordinates[i], coordinates[i + 1]) for i in range(len(coordinates) - 1)]:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color = color)


def draw_tangents(xs: [], ys: []):
    for x in xs:
        plt.plot([x, x], [0, 1], linestyle = ':', color = 'lightgray')
    for y in ys:
        plt.plot([0, 1], [y, y], linestyle = ':', color = 'lightgray')


def show():
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    plt.xlabel('$x_1$', fontsize = 16)
    plt.ylabel('$x_2$', fontsize = 16)
    plt.show()


def draw_mids(xs: [], ys: []):
    xm = map(lambda xt: (xt[0] + xt[1]) / 2, [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)])
    ym = map(lambda yt: (yt[0] + yt[1]) / 2, [(ys[i], ys[i + 1]) for i in range(len(ys) - 1)])
    cross_join = list(zip(*itertools.product(xm, ym)))
    plt.scatter(cross_join[0], cross_join[1], zorder = 0)


def fill_with_neighbors(xs: [], ys: [], indexes: [], c1 = 'y', c2 = 'b'):
    plt.fill(
        [xs[indexes[0]], xs[indexes[0]], xs[indexes[0] + 1], xs[indexes[0] + 1]],
        [ys[indexes[1]], ys[indexes[1] + 1], ys[indexes[1] + 1], ys[indexes[1]]],
        color = c1,
        zorder = -1
    )
    for i in range(indexes[0] - 1, indexes[0] + 2):
        for j in range(indexes[1] - 1, indexes[1] + 2):
            if i < 0 or j < 0 or i >= (len(xs) - 1) or j >= (len(ys) - 1) or (i == indexes[0] and j == indexes[1]): continue
            plt.fill(
                [xs[i], xs[i], xs[i + 1], xs[i + 1]],
                [ys[j], ys[j + 1], ys[j + 1], ys[j]],
                color = c2,
                zorder = -1
            )


def fill_two_rings(xs: [], ys: [], indexes: [], c1 = 'y', c2 = 'b', c3 = 'r'):
    plt.fill(
        [xs[indexes[0]], xs[indexes[0]], xs[indexes[0] + 1], xs[indexes[0] + 1]],
        [ys[indexes[1]], ys[indexes[1] + 1], ys[indexes[1] + 1], ys[indexes[1]]],
        color = c1,
        zorder = -1
    )
    for i in range(indexes[0] - 1, indexes[0] + 2):
        for j in range(indexes[1] - 1, indexes[1] + 2):
            if i < 0 or j < 0 or i >= (len(xs) - 1) or j >= (len(ys) - 1) or (i == indexes[0] and j == indexes[1]): continue
            plt.fill(
                [xs[i], xs[i], xs[i + 1], xs[i + 1]],
                [ys[j], ys[j + 1], ys[j + 1], ys[j]],
                color = c2,
                zorder = -1
            )
    for i in range(indexes[0] - 2, indexes[0] + 3):
        for j in range(indexes[1] - 2, indexes[1] + 3):
            if i < 0 or j < 0 or i >= (len(xs) - 1) or j >= (len(ys) - 1) or (i >= indexes[0] - 1 and i <= indexes[0] + 1 and j >= indexes[1] - 1 and j <= indexes[1] + 1): continue
            plt.fill(
                [xs[i], xs[i], xs[i + 1], xs[i + 1]],
                [ys[j], ys[j + 1], ys[j + 1], ys[j]],
                color = c3,
                zorder = -1
            )



### Script
tree1 = [[0, .6], [.3, .6], [.3, .2], [.7, .2], [.7, 1]]
tree2 = [[0, .4], [.6, .4], [.6, 1]]
tree3 = [[0, 0], [.2, 0], [.2, .4], [.4, .4], [.4, .8], [.9, .8], [.9, 1]]
trees = [tree1, tree2, tree3]

for (tree, c) in zip(trees, colors):
    draw_tree(tree, c)
show()


for (tree, c) in zip(trees, colors):
    draw_tree(tree, c)
points = [item for sublist in trees for item in sublist]
xs = set(map(lambda p: p[0], points))
ys = set(map(lambda p: p[1], points))
draw_tangents(xs, ys)
show()


for (tree, c) in zip(trees, colors):
    draw_tree(tree, c)
points = [item for sublist in trees for item in sublist]
xs = list(set(list(map(lambda p: p[0], points)) + [0, 1]))
ys = list(set(list(map(lambda p: p[1], points)) + [0, 1]))
xs.sort()
ys.sort()
draw_tangents(xs, ys)
fill_with_neighbors(xs, ys, [3, 2], 'darkgrey', 'gainsboro')
draw_mids(xs, ys)
show()

for (tree, c) in zip(trees, colors):
    draw_tree(tree, c)
draw_tangents(xs, ys)
fill_two_rings(xs, ys, [3, 2], 'grey', 'darkgrey', 'gainsboro')
draw_mids(xs, ys)
show()
