import unittest
from ClassifierData import ClassifierData


class ClassifierDataTest(unittest.TestCase):

    def test_validate_type_of_classifier(self):
        # given
        type_of_classifier = 'test'
        # when
        classifier_data = ClassifierData(type_of_classifier = type_of_classifier)
        # then
        with self.assertRaisesRegex(Exception, 'type_of_classifier must be of type ClfType'):
            classifier_data.validate_type_of_classifier()

    def test_validate_are_samples_generated(self):
        # given
        are_samples_generated = 'test'
        # when
        classifier_data = ClassifierData(are_samples_generated = are_samples_generated)
        # then
        with self.assertRaisesRegex(Exception, 'are_samples_generated must be of type boolean'):
            classifier_data.validate_are_samples_generated()

    def test_validate_number_of_samples_if_generated_non_int(self):
        # given
        number_of_samples_if_generated = 'test'
        # when
        classifier_data = ClassifierData(number_of_samples_if_generated = number_of_samples_if_generated)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_samples_if_generated must be of type int'):
            classifier_data.validate_number_of_samples_if_generated()

    def test_validate_number_of_samples_if_generated_too_low(self):
        # given
        number_of_samples_if_generated = ClassifierData.MINIMAL_NUMBER_OF_SAMPLES - 1
        # when
        classifier_data = ClassifierData(number_of_samples_if_generated = number_of_samples_if_generated)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_samples_if_generated must be at least'):
            classifier_data.validate_number_of_samples_if_generated()

    def test_validate_number_of_dataset_if_not_generated_non_int(self):
        # given
        number_of_dataset_if_not_generated = 'test'
        # when
        classifier_data = ClassifierData(number_of_dataset_if_not_generated = number_of_dataset_if_not_generated)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_dataset_if_not_generated must be of type int'):
            classifier_data.validate_number_of_dataset_if_not_generated()

    def test_validate_number_of_dataset_if_not_generated_too_low(self):
        # given
        number_of_dataset_if_not_generated = -1
        # when
        classifier_data = ClassifierData(number_of_dataset_if_not_generated = number_of_dataset_if_not_generated)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_dataset_if_not_generated must be positive'):
            classifier_data.validate_number_of_dataset_if_not_generated()

    def test_validate_switch_columns_while_loading(self):
        # given
        switch_columns_while_loading = 'test'
        # when
        classifier_data = ClassifierData(switch_columns_while_loading = switch_columns_while_loading)
        # then
        with self.assertRaisesRegex(Exception, 'switch_columns_while_loading must be of type boolean'):
            classifier_data.validate_switch_columns_while_loading()

    def test_validate_number_of_space_parts_non_int(self):
        # given
        number_of_space_parts = 'test'
        # when
        classifier_data = ClassifierData(number_of_space_parts = number_of_space_parts)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_space_parts must be of type int'):
            classifier_data.validate_number_of_space_parts()

    def test_validate_number_of_space_parts_too_low(self):
        # given
        number_of_space_parts = -1
        # when
        classifier_data = ClassifierData(number_of_space_parts = number_of_space_parts)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_space_parts must be positive'):
            classifier_data.validate_number_of_space_parts()

    def test_validate_number_of_classifiers_non_int(self):
        # given
        number_of_classifiers = 'test'
        # when
        classifier_data = ClassifierData(number_of_classifiers = number_of_classifiers)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_classifiers must be of type int'):
            classifier_data.validate_number_of_classifiers()

    def test_validate_number_of_classifiers_too_low(self):
        # given
        number_of_classifiers = -1
        # when
        classifier_data = ClassifierData(number_of_classifiers = number_of_classifiers)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_classifiers must be positive'):
            classifier_data.validate_number_of_classifiers()

    def test_validate_number_of_best_classifiers_non_int(self):
        # given
        number_of_best_classifiers = 'test'
        # when
        classifier_data = ClassifierData(number_of_best_classifiers = number_of_best_classifiers)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_best_classifiers must be of type int'):
            classifier_data.validate_number_of_best_classifiers()

    def test_validate_number_of_best_classifiers_too_low(self):
        # given
        number_of_best_classifiers = -1
        # when
        classifier_data = ClassifierData(number_of_best_classifiers = number_of_best_classifiers)
        # then
        with self.assertRaisesRegex(Exception, 'number_of_best_classifiers must be positive'):
            classifier_data.validate_number_of_best_classifiers()

    def test_validate_draw_color_plot(self):
        # given
        draw_color_plot = 'test'
        # when
        classifier_data = ClassifierData(show_color_plot = draw_color_plot)
        # then
        with self.assertRaisesRegex(Exception, 'draw_color_plot must be of type boolean'):
            classifier_data.validate_draw_color_plot()

    def test_validate_write_computed_scores(self):
        # given
        write_computed_scores = 'test'
        # when
        classifier_data = ClassifierData(write_computed_scores = write_computed_scores)
        # then
        with self.assertRaisesRegex(Exception, 'write_computed_scores must be of type boolean'):
            classifier_data.validate_write_computed_scores()

    def test_validate_show_plots(self):
        # given
        show_plots = 'test'
        # when
        classifier_data = ClassifierData(show_plots = show_plots)
        # then
        with self.assertRaisesRegex(Exception, 'show_plots must be of type boolean'):
            classifier_data.validate_show_plots()

    def test_validate_columns_non_matrix(self):
        # given
        columns = 'test'
        # when
        classifier_data = ClassifierData(columns = columns)
        # then
        with self.assertRaisesRegex(Exception, 'Columns must be vector of size 1 x 2'):
            classifier_data.validate_columns()

    def test_validate_columns_non_ints(self):
        # given
        columns = ['test', 'test']
        # when
        classifier_data = ClassifierData(columns = columns)
        # then
        with self.assertRaisesRegex(Exception, 'Elements of columns must be of type int'):
            classifier_data.validate_columns()

    def test_validate_columns_non_positive(self):
        # given
        columns = [-1, -1]
        # when
        classifier_data = ClassifierData(columns = columns)
        # then
        with self.assertRaisesRegex(Exception, 'Column number must be positive'):
            classifier_data.validate_columns()

    def test_validate_columns_not_different(self):
        # given
        columns = [1, 1]
        # when
        classifier_data = ClassifierData(columns = columns)
        # then
        with self.assertRaisesRegex(Exception, 'Elements of matrix must be different'):
            classifier_data.validate_columns()

    def test_validate_is_validation_hard(self):
        # given
        is_validation_hard = 'test'
        # when
        classifier_data = ClassifierData(is_validation_hard = is_validation_hard)
        # then
        with self.assertRaisesRegex(Exception, 'is_validation_hard must be of type boolean'):
            classifier_data.validate_is_validation_hard()

    def test_validate_filename(self):
        # given
        filename = 0
        # when
        classifier_data = ClassifierData(filename = filename)
        # then
        with self.assertRaisesRegex(Exception, 'filename must be of type str'):
            classifier_data.validate_filename()

    def test_validate_generate_all_permutations(self):
        # given
        generate_all_permutations = 'test'
        # when
        classifier_data = ClassifierData(generate_all_permutations = generate_all_permutations)
        # then
        with self.assertRaisesRegex(Exception, 'generate_all_permutations must be of type bool'):
            classifier_data.validate_generate_all_permutations()


if __name__ == '__main__':
    unittest.main()
