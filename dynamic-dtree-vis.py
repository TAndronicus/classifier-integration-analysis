import matplotlib.pyplot as plt

### Constants
colors = ['r', 'g', 'b']


### Functions
def draw_tree(coordinates: [], color='r'):
    for (p1, p2) in [(coordinates[i], coordinates[i + 1]) for i in range(len(coordinates) - 1)]:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color)


def draw_tangents(xs: [], ys: []):
    for x in xs:
        plt.plot([x, x], [0, 1], linestyle = ':', color = 'lightgray')
    for y in ys:
        plt.plot([0, 1], [y, y], linestyle = ':', color = 'lightgray')


def show():
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    plt.show()


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
