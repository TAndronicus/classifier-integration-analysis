from Orange.evaluation import graph_ranks
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 12})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex = True)

critical_value = 1.66
algorithms = {
    # @formatter:off
    '$\Psi_{mv}$':       5.46,
    '$\Psi_{rf}$':       2.29,
    '$\Psi_{vol}^{20}$': 4.72,
    '$\Psi_{vol}^{40}$': 4.31,
    '$\Psi_{vol}^{60}$': 5.46
    # @formatter:on
}
avranks = [*algorithms.values()]
names = [*algorithms.keys()]
# algorithms = sorted(algorithms, key=algorithms.get)
graph_ranks(avranks, names, cd = critical_value, textspace = 1.5)
plt.show()
