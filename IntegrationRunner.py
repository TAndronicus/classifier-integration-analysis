import MergingAlgorithm
import ClassifLibrary
import FileHelper
from NotEnoughSamplesError import NotEnoughSamplesError
import os
from ClfType import ClfType
from CompositionType import CompositionType
from datetime import datetime

filenames = ['biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx', 'data_banknote_authentication.csv',
             'haberman.dat', 'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv', 'seismic_bumps.dat',
             'twonorm.dat', 'wdbc.dat', 'wisconsin.dat']
type_of_classifier = ClfType.LINEAR
are_samples_generated = False
number_of_samples_if_generated = 1000
number_of_dataset_if_not_generated = 0
draw_color_plot = False
write_computed_scores = False
show_plots = False
show_only_first_plot = True
is_validation_hard = False
generate_all_permutations = False
bagging = True
type_of_composition = CompositionType.MEDIAN

results_directory_relative = 'results'
logging_to_file = True

files_to_switch = ['haberman.dat', 'sonar.dat']
numbers_of_base_classifiers = list(range(3, 4))
space_division = [3]

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
for number_of_base_classifiers in numbers_of_base_classifiers:
    print('Number of classifiers: ', number_of_base_classifiers)
    results_pro_classifier = []
    for number_of_space_parts in space_division:
        print('Number of space parts:', number_of_space_parts)
        results_pro_division = []
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
                                              number_of_space_parts = number_of_space_parts,
                                              number_of_classifiers = number_of_base_classifiers,
                                              number_of_best_classifiers = number_of_base_classifiers - 1,
                                              show_color_plot = draw_color_plot,
                                              write_computed_scores = write_computed_scores,
                                              show_plots = show_plots,
                                              show_only_first_plot = show_only_first_plot,
                                              is_validation_hard = is_validation_hard,
                                              filename = 'datasets//' + filename,
                                              generate_all_permutations = generate_all_permutations,
                                              log_number = log_number,
                                              bagging = bagging,
                                              logging_to_file = logging_to_file)
            try:
                mv_score, merged_score, mv_mcc, merged_mcc = MergingAlgorithm.run(classifier_data)
                results_pro_division.append([mv_score, merged_score, mv_mcc, merged_mcc])
            except NotEnoughSamplesError as e:
                print(e.args[0])
                mv_score, merged_score, mv_mcc, merged_mcc = float('nan'), float('nan'), float('nan'), float('nan')
        results_pro_classifier.append(results_pro_division)
    results.append(results_pro_classifier)
FileHelper.save_merging_results_pro_space_division_pro_base_classif(filenames, results, numbers_of_base_classifiers,
                                                                    space_division,
                                                                    result_filename = results_directory_relative +
                                                                                      '//Results' +
                                                                                      str(result_file_number) + '.xls')

log = open(results_directory_relative + '//integration' + str(log_number) + '.log', 'a')
log.write('Finishing algorithm: ' + str(datetime.now()))
log.close()
