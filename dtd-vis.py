import matplotlib.pyplot as plt

def divide_plot(div):
    for i in range(1, div + 1):
        cutpoint = i / (div + 1)
        plt.plot([0, 1], [cutpoint, cutpoint], 'k--')
        plt.plot([cutpoint, cutpoint], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])


# Division
divide_plot(3)
plt.show()

# Decision tree
x_cut, y_cut = .6, .7
plt.plot([x_cut, x_cut], [0, 1], 'k')
plt.plot([0, 1], [y_cut, y_cut], 'k')
plt.axis([0, 1, 0, 1])
plt.fill([0, 0, x_cut, x_cut], [0, y_cut, y_cut, 0], 'b', alpha=.2)
plt.fill([x_cut, x_cut, 1, 1], [y_cut, 1, 1, y_cut], 'b', alpha=.2)
plt.fill([0, 0, x_cut, x_cut], [y_cut, 1, 1, y_cut], 'r', alpha=.4)
plt.fill([x_cut, x_cut, 1, 1], [0, y_cut, y_cut, 0], 'r', alpha=.4)
plt.show()

# Subregion - subspace bind
divide_plot(2)
x_min, x_max, y_min, y_max = .0, .6, .1, .5
plt.fill([x_min, x_min, x_max, x_max], [y_min, y_max, y_max, y_min], 'b', alpha=.2)
plt.fill(
    [x_min, x_min, x_max, x_max, x_min, x_min, 1, 1],
    [0, y_min, y_min, y_max, y_max, 1, 1, 0],
    'r', alpha=.4
)
plt.plot([.5], [1 / 6], 'ko')
plt.show()

# Integration
divide_plot(2)
x_min, x_max, y_min, y_max = .4, .6, .1, .5
plt.fill([x_min, x_min, x_max, x_max], [y_min, y_max, y_max, y_min], 'b', alpha=.2)
plt.fill(
    [x_min, x_min, x_max, x_max, x_min, x_min, 1, 1],
    [0, y_min, y_min, y_max, y_max, 1, 1, 0],
    'r', alpha=.4
)
plt.fill([0, 0, x_min, x_min], [0, 1, 1, 0], 'r', alpha=.4)
plt.plot([.5], [1 / 6], 'ko')
plt.show()

divide_plot(2)
x_min, x_max, y_min, y_max = .2, .6, .1, .7
plt.fill([x_min, x_min, x_max, x_max], [y_min, y_max, y_max, y_min], 'r', alpha=.4)
plt.fill(
    [x_min, x_min, x_max, x_max, x_min, x_min, 1, 1],
    [0, y_min, y_min, y_max, y_max, 1, 1, 0],
    'b', alpha=.2
)
plt.fill([0, 0, x_min, x_min], [0, 1, 1, 0], 'b', alpha=.2)
plt.plot([.5], [1 / 6], 'ko')
plt.show()

divide_plot(2)
plt.fill(
    [1 / 3, 1 / 3, 2 / 3, 2 / 3],
    [0, 1 / 3, 1 / 3, 0],
    'r', alpha=.4
)
plt.show()

# Example for not 1NN
x, y = .7, .3
plt.fill(
    [0, 0, 1, 1],
    [0, 1, 1, 0],
    'b', alpha = .2
)
plt.fill(
    [x, x, 1, 1],
    [0, y, y, 0],
    'r', alpha = .4
)
plt.plot(
    [0, 1],
    [y, y],
    'k--'
)
plt.plot(
    [x, x],
    [0, 1],
    'k--'
)
plt.plot(x / 2, y / 2, 'bo')
plt.plot(x / 2, (y + 1) / 2, 'bo')
plt.plot((x + 1) / 2, (y + 1) / 2, 'bo')
plt.plot((x + 1) / 2, y / 2, 'ro')
plt.plot(x - .05, y - .05, 'ko')
plt.show()
