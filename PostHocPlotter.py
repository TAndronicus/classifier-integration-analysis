import os

import matplotlib.pyplot as plt
from Orange.evaluation import graph_ranks
from matplotlib import rc

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 15})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex = True)
critical_value = 0.75
results_catalog = 'reports/'
results_filename = 'res-dts.csv'
figures_catalog = 'figures'
fig_extension = 'png'
save_to_files = True

absolute_path = os.path.join(os.path.dirname(__file__), results_catalog + results_filename)
experiments = []
algorithm = {}
previous_matched = False
titles = []
with(open(absolute_path)) as file:
    for line in file.readlines():
        if(line.startswith('n_clf')):
            parts = line.split(',')
            titles.append('_'.join(['_'.join(x.split(':').strip()) for x in parts]))
        elif(line.startswith('$')):
            parts = line.split(',')
            algorithm[parts[0]] = float(parts[-1])
            previous_matched = True
        elif (previous_matched):
            experiments.append(algorithm)
            algorithm = {}
            previous_matched = False

'''
algorithms = {
    '$\Psi_{mv}$': 2,
    '$\Psi_{rf}$': 2.14,
    '$\Psi_{i}$': 1.61
}
'''
for algorithm, title in zip(experiments, titles):
    avranks = [*algorithm.values()]
    names = [*algorithm.keys()]
    graph_ranks(avranks, names, cd = critical_value, textspace = 1.5)
    if (save_to_files):
        plt.savefig(figures_catalog + '/' + title + '.' + fig_extension)
    else:
        plt.show()
