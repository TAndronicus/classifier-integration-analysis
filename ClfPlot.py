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
    # '$\Psi_{mv}$':       5.46,
    '$\Psi_{rf}$':  3.76,
    '$\Psi_{0.0}$': 3.65,
    '$\Psi_{0.3}$': 4.56,
    '$\Psi_{0.7}$': 5.36,
    '$\Psi_{1.0}$': 4.90
    # @formatter:on
}
avranks = [*algorithms.values()]
names = [*algorithms.keys()]
# algorithms = sorted(algorithms, key=algorithms.get)
graph_ranks(avranks, names, cd = critical_value, textspace = 1.5)
plt.show()
