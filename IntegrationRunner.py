import MergingAlgorithm
import ClassifLibrary
import FileHelper

filenames = ['appendicitis.dat', 'biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx', 'data_banknote_authentication.csv',
             'haberman.dat', 'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv', 'seismic_bumps.dat', 'sonar.dat',
             'spectfheart.dat', 'twonorm.dat', 'wdbc.dat', 'wisconsin.dat']
type_of_classifier = ClassifLibrary.ClfType.LINEAR
are_samples_generated = False
number_of_samples_if_generated = 10000
number_of_dataset_if_not_generated = 0
columns = [0, 1]
number_of_space_parts = 5
number_of_classifiers = 3
number_of_best_classifiers = number_of_classifiers - 1
draw_color_plot = False
write_computed_scores = False
show_plots = False
is_validation_hard = False
generate_all_permutations = True
files_to_switch = ['haberman.dat', 'seismic_bumps.dat', 'sonar.dat']

results = []
for filename in filenames:
    print('Analysing ' + filename)
    if files_to_switch.__contains__(filename):
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
                                      number_of_classifiers = number_of_classifiers,
                                      number_of_best_classifiers = number_of_best_classifiers,
                                      show_color_plot = draw_color_plot, write_computed_scores = write_computed_scores,
                                      show_plots = show_plots, columns = columns,
                                      is_validation_hard = is_validation_hard,
                                      filename = 'datasets//' + filename,
                                      generate_all_permutations = generate_all_permutations)

    mv_score, merged_score, mv_mcc, merged_mcc = MergingAlgorithm.run(classifier_data)
    results.append([mv_score, merged_score, mv_mcc, merged_mcc])
FileHelper.save_merging_results(filenames, results)
