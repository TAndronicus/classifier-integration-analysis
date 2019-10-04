import os

from DtdBatchRes import DtdBatchRes

filenames = ['bi', 'bu', 'c', 'd', 'h', 'i', 'm', 'p', 'se', 'wd', 'wi']
n_divs = [20, 40, 60]
path = os.path.join(os.path.dirname(__file__), 'partial-results')

first_run, lines_length, summary_values = True, [], []


file_counter = 0
for partial_file_name in os.listdir(path):
    with(open(os.path.join(path, partial_file_name))) as file:
        line_counter = 0
        for line in file.readlines():
            if line.isspace(): continue
            values = [float(item) for item in line.split(',')]
            if file_counter == 0:
                summary_values.append(values)
            else:
                summary_values[line_counter] = [summary_values[line_counter][index] + values[index] for index in range(len(values))]
            line_counter = line_counter + 1
        file_counter = file_counter + 1

summary_values = [[value / file_counter for value in row] for row in summary_values]

with(open(os.path.join(path, 'result'), 'w')) as file:
    for row in summary_values:
        for value in row:
            file.write(str(value) + ',')
        file.write('\n')
