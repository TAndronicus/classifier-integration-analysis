import unittest
from ClassifierData import ClassifierData
from MergingAlgorithm import run


class MergingAlgorithmTest(unittest.TestCase):

    def test_should_return_no_error_on_default_merging_algorithm(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run()
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_with_no_permutations(self):
        # given
        generate_all_permutations = False
        classifier_data = ClassifierData(generate_all_permutations = generate_all_permutations)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_switching_columns(self):
        # given
        switch_columns_while_loading = True
        classifier_data = ClassifierData(switch_columns_while_loading = switch_columns_while_loading)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_with_no_permutations_switching_columns(self):
        # given
        generate_all_permutations = False
        switch_columns_while_loading = True
        classifier_data = ClassifierData(generate_all_permutations = generate_all_permutations,
                                         switch_columns_while_loading = switch_columns_while_loading)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat')
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv')
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv')
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv')
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_show_plot(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(ClassifierData(show_plots = True))
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat', show_plots = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv', show_plots = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv', show_plots = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv', show_plots = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(ClassifierData(show_plots = True, show_color_plot = True))
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat', show_plots = True, show_color_plot = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv', show_plots = True, show_color_plot = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv', show_plots = True,
                                         show_color_plot = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots_builtin(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv', show_plots = True, show_color_plot = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin_computed(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(ClassifierData(show_plots = True, show_color_plot = True,
                                                                        write_computed_scores = True))
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'appendicitis.dat', show_plots = True, show_color_plot = True,
                                         write_computed_scores = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'biodeg.scsv', show_plots = True, show_color_plot = True,
                                         write_computed_scores = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'data_banknote_authentication.csv', show_plots = True,
                                         show_color_plot = True, write_computed_scores = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots_builtin_computed(self):
        # given
        classifier_data = ClassifierData(filename = 'pop_failures.tsv', show_plots = True, show_color_plot = True,
                                         write_computed_scores = True)
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)


if __name__ == '__main__':
    unittest.main()
