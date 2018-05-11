import matplotlib
matplotlib.use('Agg')
import unittest
import math
from ClassifierData import ClassifierData
from ClfType import ClfType
from CompositionType import CompositionType
from MergingAlgorithm import run


class MergingAlgorithmtest(unittest.TestCase):
    """Suite of integration tests for merging (integrating) algorithm

    """

    def test_should_return_no_error_on_default_merging_algorithm(self):
        # given
        logging_to_file = False
        classifier_data = ClassifierData(logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_mean_classifier(self):
        # given
        type_of_classifier = ClfType.MEAN
        logging_to_file = False
        classifier_data = ClassifierData(type_of_classifier = type_of_classifier,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_mean_classifier_showing_plot(self):
        # given
        type_of_classifier = ClfType.MEAN
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(type_of_classifier = type_of_classifier,
                                         show_plots = show_plots,
                                         logging_to_file = logging_to_file,
                                         generate_all_permutations = False)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_with_no_permutations(self):
        # given
        generate_all_permutations = False
        logging_to_file = False
        classifier_data = ClassifierData(generate_all_permutations = generate_all_permutations,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_switching_columns(self):
        # given
        switch_columns_while_loading = True
        logging_to_file = False
        classifier_data = ClassifierData(switch_columns_while_loading = switch_columns_while_loading,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_with_no_permutations_switching_columns(self):
        # given
        generate_all_permutations = False
        switch_columns_while_loading = True
        logging_to_file = False
        classifier_data = ClassifierData(generate_all_permutations = generate_all_permutations,
                                         switch_columns_while_loading = switch_columns_while_loading,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file(self):
        # given
        filename = 'appendicitis.dat'
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename, logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file(self):
        # given
        filename = 'biodeg.scsv'
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename, logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file(self):
        # given
        filename = 'data_banknote_authentication.csv'
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file(self):
        # given
        logging_to_file = False
        classifier_data = ClassifierData(filename = 'pop_failures[0][0].tsv', logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_show_plot(self):
        # given
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(show_plots = show_plots,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots(self):
        # given
        filename = 'appendicitis.dat'
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots(self):
        # given
        filename = 'biodeg.scsv'
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots(self):
        # given
        filename = 'data_banknote_authentication.csv'
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots(self):
        # given
        filename = 'pop_failures[0][0].tsv'
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin(self):
        # given
        show_plots = True
        show_color_plot = True
        logging_to_file = False
        classifier_data = ClassifierData(show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots_builtin(self):
        # given
        filename = 'appendicitis.dat'
        show_plots = True
        show_color_plot = True
        logging_to_file = False
        classifier_data = ClassifierData(filename= filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots_builtin(self):
        # given
        filename = 'biodeg.scsv'
        show_plots = True
        show_color_plot = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots_builtin(self):
        # given
        filename = 'data_banknote_authentication.csv'
        show_plots = True
        show_color_plot = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots_builtin(self):
        # given
        filename = 'pop_failures[0][0].tsv'
        show_plots = True
        show_color_plot = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_plots_builtin_computed(self):
        # given
        show_plots = True
        show_color_plot = True
        write_computed_scores = True
        logging_to_file = False
        classifier_data = ClassifierData(show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_dat_file_plots_builtin_computed(self):
        # given
        filename = 'appendicitis.dat'
        show_plots = True
        show_color_plot = True
        write_computed_scores = True
        logging_too_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_too_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_scsv_file_plots_builtin_computed(self):
        # given
        filename = 'biodeg.scsv'
        show_plots = True
        show_color_plot = True
        write_computed_scores = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_csv_file_plots_builtin_computed(self):
        # given
        filename = 'data_banknote_authentication.csv'
        show_plots = True
        show_color_plot = True
        write_computed_scores = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_on_default_merging_algorithm_reading_tsv_file_plots_builtin_computed(self):
        # given
        filename = 'pop_failures[0][0].tsv'
        show_plots = True
        show_color_plot = True
        write_computed_scores = True
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         show_plots = show_plots,
                                         show_color_plot = show_color_plot,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_right_value_on_mock_test_case(self):
        # given
        filename = 'TestCases.xlsx'
        are_samples_generated = False
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         are_samples_generated = are_samples_generated,
                                         logging_to_file = logging_to_file)
        TP = 500
        TN = 400
        FP = 70
        FN = 30
        expected_score = (TP + TN) / (TP + TN + FP + FN)
        expected_mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertAlmostEqual(expected_score, mv_score, delta = .01)
        self.assertAlmostEqual(expected_score, i_score, delta = .02)
        self.assertAlmostEqual(expected_mcc, mv_mcc, delta = .02)
        self.assertAlmostEqual(expected_mcc, i_mcc, delta = .04)

    def test_should_return_right_value_on_mock_test_case_one_iter(self):
        # given
        filename = 'TestCases.xlsx'
        are_samples_generated = False
        generate_all_permutations = False
        logging_to_file = False
        classifier_data = ClassifierData(filename = filename,
                                         are_samples_generated = are_samples_generated,
                                         generate_all_permutations = generate_all_permutations,
                                         logging_to_file = logging_to_file)
        TP = 500
        TN = 400
        FP = 70
        FN = 30
        expected_score = (TP + TN) / (TP + TN + FP + FN)
        expected_mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FN) * (TN + FP))
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertAlmostEqual(expected_score, mv_score, delta = .01)
        self.assertAlmostEqual(expected_score, i_score, delta = .03)
        self.assertAlmostEqual(expected_mcc, mv_mcc, delta = .01)
        self.assertAlmostEqual(expected_mcc, i_mcc, delta = .05)

    def test_should_return_no_error_on_default_merging_algorithm_with_bagging(self):
        # given
        bagging = True
        logging_to_file = False
        classifier_data = ClassifierData(bagging = bagging,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_with_bagging_on_generated_samples(self):
        # given
        are_samples_generated = True
        generate_all_permutations = False
        bagging = True
        logging_to_file = False
        classifier_data = ClassifierData(are_samples_generated = are_samples_generated,
                                         generate_all_permutations = generate_all_permutations,
                                         bagging = bagging,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_mean_classif_when_scores_computed_manually(self):
        # given
        type_of_classifier = ClfType.MEAN
        generate_all_permutations = False
        write_computed_scores = True
        logging_to_file = False
        classifier_data = ClassifierData(type_of_classifier = type_of_classifier,
                                         generate_all_permutations = generate_all_permutations,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_median_classif_integr(self):
        # given
        type_of_composition = CompositionType.MEDIAN
        logging_to_file = False
        classifier_data = ClassifierData(type_of_composition = type_of_composition,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_median_classif_integrfor_mean_classifier(self):
        # given
        type_of_classifier = ClfType.MEAN
        type_of_composition = CompositionType.MEDIAN
        logging_to_file = False
        classifier_data = ClassifierData(type_of_classifier = type_of_classifier,
                                         type_of_composition = type_of_composition,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_median_classif_integr_writing_manual(self):
        # given
        type_of_composition = CompositionType.MEDIAN
        write_computed_scores = True
        logging_to_file = False
        classifier_data = ClassifierData(type_of_composition = type_of_composition,
                                         write_computed_scores = write_computed_scores,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)

    def test_should_return_no_error_for_median_classif_integr_showing_plots(self):
        # given
        show_plots = True
        logging_to_file = False
        classifier_data = ClassifierData(show_plots = show_plots,
                                         logging_to_file = logging_to_file)
        # when
        res = run(classifier_data)
        mv_score = res[0][0].mv_score
        mv_mcc = res[0][0].mv_mcc
        i_score = res[0][0].i_score
        i_mcc = res[0][0].i_mcc
        # then
        self.assertIsNotNone(mv_score)
        self.assertIsNotNone(i_score)
        self.assertIsNotNone(mv_mcc)
        self.assertIsNotNone(i_mcc)
        if mv_score > .5:
            self.assertTrue(mv_mcc >= 0)
        elif mv_score < .5:
            self.assertTrue(mv_mcc <= 0)
        if i_score > .5:
            self.assertTrue(i_mcc >= 0)
        elif i_score < .5:
            self.assertTrue(i_mcc <= 0)


if __name__ == '__main__':
    unittest.main()
