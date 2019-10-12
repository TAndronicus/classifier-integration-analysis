from Orange.evaluation import graph_ranks
import matplotlib.pyplot as plt

critical_value = .5
algorithms = {
    'mv': 4.2,
    'rf': 5,
    'nref': 4.9
}
avranks = [*algorithms.values()]
names = [*algorithms.keys()]
# algorithms = sorted(algorithms, key=algorithms.get)
graph_ranks(avranks, names, cd=.5, width=6, textspace=1.5)
plt.show()

