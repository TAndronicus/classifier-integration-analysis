import unittest
import math
from ClassifierData import ClassifierData
from MergingAlgorithm import run


class MergingAlgorithmtest(unittest.TestCase):
    """Suite of integration tests for merging (integrating) algorithm

    """

    def test_should_return_no_error_on_default_merging_algorithm(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run()
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_show_plot(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(ClassifierData(show_plots = True))
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin(self):
        # given
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(ClassifierData(show_plots = True, show_color_plot = True))
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(merged_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(merged_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

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
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if merged_score > .5:
            self.assertTrue(merged_mcc >= 0)
        elif merged_score < .5:
            self.assertTrue(merged_mcc <= 0)

    def test_should_return_right_value_on_mock_test_case(self):
        # given
        filename = 'TestCases.xlsx'
        are_samples_generated = False
        classifier_data = ClassifierData(filename = filename, are_samples_generated = are_samples_generated)
        TP = 500
        TN = 400
        FP = 70
        FN = 30
        expected_score = (TP + TN) / (TP + TN + FP + FN)
        expected_mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertAlmostEqual(expected_score, mv_score, delta = .01)
        self.assertAlmostEqual(expected_score, merged_score, delta = .02)
        self.assertAlmostEqual(expected_mcc, mv_mcc, delta = .02)
        self.assertAlmostEqual(expected_mcc, merged_mcc, delta = .04)

    def test_should_return_right_value_on_mock_test_case_one_iter(self):
        # given
        filename = 'TestCases.xlsx'
        are_samples_generated = False
        generate_all_permutations = False
        classifier_data = ClassifierData(filename = filename, are_samples_generated = are_samples_generated,
                                         generate_all_permutations = generate_all_permutations)
        TP = 500
        TN = 400
        FP = 70
        FN = 30
        expected_score = (TP + TN) / (TP + TN + FP + FN)
        expected_mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))
        # when
        mv_score, merged_score, mv_mcc, merged_mcc = run(classifier_data)
        # then
        self.assertAlmostEqual(expected_score, mv_score, delta = .01)
        self.assertAlmostEqual(expected_score, merged_score, delta = .03)
        self.assertAlmostEqual(expected_mcc, mv_mcc, delta = .01)
        self.assertAlmostEqual(expected_mcc, merged_mcc, delta = .05)


if __name__ == '__main__':
    unittest.main()
