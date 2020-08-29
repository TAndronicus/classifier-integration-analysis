import matplotlib.pyplot as plt
from Orange.evaluation import graph_ranks
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 15})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex = True)

critical_value = 0.38
algorithms = {
    # @formatter:off
    '$\Psi_{mv}$': 2,
    '$\Psi_{rf}$': 2.14,
    '$\Psi_{i}$': 1.61
    # @formatter:on
}
avranks = [*algorithms.values()]
names = [*algorithms.keys()]
# algorithms = sorted(algorithms, key=algorithms.get)
graph_ranks(avranks, names, cd = critical_value, textspace = 1.5)
plt.show()
