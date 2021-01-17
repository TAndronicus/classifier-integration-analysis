import os

import matplotlib.pyplot as plt
from Orange.evaluation import graph_ranks
from matplotlib import rc

'''IMPORTANT: file must end with an empty line'''

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 15})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
# critical_value = 1.18
# critical_value = 1.44
# critical_value = 0.38
critical_value = 0.75
results_catalog = 'reports/'
# results_filename = '1-displacement.csv'
# results_filename = '2-batch.csv'
# results_filename = '3-dynamic-dtree.csv'
results_filename = '4-dynamic-ring.csv'
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
        if (line.startswith('n_')):
            parts = line.split(',')
            titles.append('_'.join(['_'.join([y.strip() for y in x.split(':')]) for x in parts]))
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
