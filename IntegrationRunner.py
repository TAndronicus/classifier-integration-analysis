import MergingAlgorithm
import ClassifLibrary
import FileHelper
from NotEnoughSamplesError import NotEnoughSamplesError
import os
from ClfType import ClfType
from CompositionType import CompositionType
from datetime import datetime

### Dataset ###
filenames = ['bi', 'bu', 'c', 'd', 'h', 'i', 'm', 'p', 's', 't', 'wd', 'wi']
filenames = FileHelper.prepare_filenames(filenames)
filenames = FileHelper.sort_filenames_by_size(filenames)
files_to_switch = ['haberman.dat', 'sonar.dat']
number_of_dataset_if_not_generated = 0

### Classification strategy ###
type_of_classifier = ClfType.LINEAR
type_of_composition = CompositionType.MEAN
is_validation_hard = False
generate_all_permutations = True
bagging = True
number_of_bagging_repetitions = 3
space_division = list(range(3, 11))
number_of_classifiers = 9

### Samples generation ###
are_samples_generated = False
number_of_samples_if_generated = 1000

### Plotting + builtin
draw_color_plot = False
write_computed_scores = False
show_plots = False
show_only_first_plot = True

### Logging ###
results_directory_relative = 'results'
logging_to_file = True
logging_intermediate_results = False

results_directory_absolute = os.path.join(os.path.dirname(__file__), results_directory_relative)
try:
    os.makedirs(results_directory_absolute)
    print('Created results directory: ', results_directory_absolute)
except FileExistsError:
    pass

log_number = 0
while True:
    if not os.path.isfile(results_directory_relative + '//integration' + str(log_number) + '.log'):
        break
    log_number += 1
log = open(results_directory_relative + '//integration' + str(log_number) + '.log', 'w')
log.write('Starting algorithm: ' + str(datetime.now()) + '\n\n')
log.close()

result_file_number = 0
while True:
    if not os.path.isfile(results_directory_relative + '//Results' + str(result_file_number) + '.xls'):
        break
    result_file_number += 1

results = []
for filename in filenames:
    print('Analysing ' + filename)
    if filename in files_to_switch:
        switch_columns_while_loading = True
        print('Switching columns')
    else:
        switch_columns_while_loading = False
    classifier_data = \
        ClassifLibrary.ClassifierData(type_of_classifier = type_of_classifier,
                                      are_samples_generated = are_samples_generated,
                                      number_of_samples_if_generated = number_of_samples_if_generated,
                                      number_of_dataset_if_not_generated = number_of_dataset_if_not_generated,
                                      switch_columns_while_loading = switch_columns_while_loading,
                                      number_of_classifiers = number_of_classifiers,
                                      show_color_plot = draw_color_plot,
                                      write_computed_scores = write_computed_scores,
                                      show_plots = show_plots,
                                      show_only_first_plot = show_only_first_plot,
                                      is_validation_hard = is_validation_hard,
                                      filename = 'datasets//' + filename,
                                      generate_all_permutations = generate_all_permutations,
                                      log_number = log_number,
                                      bagging = bagging,
                                      type_of_composition = type_of_composition,
                                      logging_to_file = logging_to_file,
                                      logging_intermediate_results = logging_intermediate_results,
                                      space_division = space_division)
    try:
        if bagging == True:
            bagging_results = []
            for i in range(number_of_bagging_repetitions):
                print('{}. bagging iteration'.format(i + 1))
                bagging_res = MergingAlgorithm.run(classifier_data)
                bagging_results.append(bagging_res)
            res = ClassifLibrary.get_mean_res(bagging_results)

        else:
            res = MergingAlgorithm.run(classifier_data)
    except NotEnoughSamplesError as e:
        print(e.args[0])
        break
    results.append(res)
FileHelper.save_res_objects_pro_space_division_pro_base_classif_with_classif_data_name(filenames, results,
                                                                                       number_of_classifiers,
                                                                                       results_directory_relative =
                                                                                       results_directory_relative,
                                                                                       classifier_data =
                                                                                       classifier_data)

log = open(results_directory_relative + '//integration' + str(log_number) + '.log', 'a')
log.write('Finishing algorithm: ' + str(datetime.now()))
log.close()
