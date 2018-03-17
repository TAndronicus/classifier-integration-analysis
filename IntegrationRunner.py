import MergingAlgorithm
import ClassifLibrary

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
show_plots = True
is_validation_hard = False

for filename in filenames:
    classifier_data = \
        ClassifLibrary.ClassifierData(type_of_classifier = type_of_classifier,
                                      are_samples_generated = are_samples_generated,
                                      number_of_samples_if_generated = number_of_samples_if_generated,
                                      number_of_dataset_if_not_generated = number_of_dataset_if_not_generated,
                                      number_of_space_parts = number_of_space_parts,
                                      number_of_classifiers = number_of_classifiers,
                                      number_of_best_classifiers = number_of_best_classifiers,
                                      show_color_plot = draw_color_plot, write_computed_scores = write_computed_scores,
                                      show_plots = show_plots, columns = columns,
                                      is_validation_hard = is_validation_hard,
                                      filename = 'datasets//' + filename)

    mv_score, merged_score, mv_mcc, merged_mcc = MergingAlgorithm.run(classifier_data)