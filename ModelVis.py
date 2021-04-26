import matplotlib.pyplot as plt

colors = {
    0: 'b',
    1: 'g'
}

with(open("models/ap")) as f:
    for line in f:
        values = list(map(lambda x: float(x), line.split(",")))
        mins, maxes, label = values[:int(len(values) / 2)], values[int(len(values) / 2):-1], int(values[-1])
        plt.fill(
            [mins[0], mins[0], maxes[0], maxes[0]],
            [mins[1], maxes[1], maxes[1], mins[1]],
            color = colors.get(label)
        )
    plt.show()
