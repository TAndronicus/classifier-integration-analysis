from ClfType import ClfType


class ClassifierData:
    """Clas to produce parameter object for classification

    """

    MINIMAL_NUMBER_OF_SAMPLES = 10

    def __init__(self, type_of_classifier: ClfType = ClfType.LINEAR, are_samples_generated: bool = True,
                 number_of_samples_if_generated: int = 1000, number_of_dataset_if_not_generated: int = 0,
                 switch_columns_while_loading: bool = False, number_of_space_parts: int = 5,
                 number_of_classifiers: int = 3, number_of_best_classifiers: int = 2, show_color_plot: bool = False,
                 write_computed_scores: bool = False, show_plots: bool = False, show_only_first_plot: bool = True,
                 columns: [] = [0, 1], is_validation_hard: bool = False, filename: str = 'new-datasets.xlsx',
                 generate_all_permutations: bool = True, log_number: int = 0):
        self.type_of_classifier = type_of_classifier
        self.are_samples_generated = are_samples_generated
        self.number_of_samples_if_generated = number_of_samples_if_generated
        self.number_of_dataset_if_not_generated = number_of_dataset_if_not_generated
        self.switch_columns_while_loading = switch_columns_while_loading
        self.number_of_space_parts = number_of_space_parts
        self.number_of_classifiers = number_of_classifiers
        self.number_of_best_classifiers = number_of_best_classifiers
        self.draw_color_plot = show_color_plot
        self.write_computed_scores = write_computed_scores
        self.show_plots = show_plots
        self.show_only_first_plot = show_only_first_plot
        self.columns = columns
        self.is_validation_hard = is_validation_hard
        self.filename = filename
        self.generate_all_permutations = generate_all_permutations
        self.log_number = log_number

    def validate(self):
        print('Validating parameters')
        self.validate_type_of_classifier()
        self.validate_are_samples_generated()
        self.validate_number_of_samples_if_generated()
        self.validate_number_of_dataset_if_not_generated()
        self.validate_switch_columns_while_loading()
        self.validate_number_of_space_parts()
        self.validate_number_of_classifiers()
        self.validate_number_of_best_classifiers()
        self.validate_draw_color_plot()
        self.validate_write_computed_scores()
        self.validate_show_plots()
        self.validate_show_only_first_plot()
        self.validate_columns()
        self.validate_is_validation_hard()
        self.validate_filename()
        self.validate_generate_all_permutations()
        self.validate_log_number()
        print('Parameters valid\n')

    def validate_type_of_classifier(self):
        if not type(self.type_of_classifier) is ClfType:
            raise Exception('type_of_classifier must be of type ClfType')

    def validate_are_samples_generated(self):
        if not type(self.are_samples_generated) is bool:
            raise Exception('are_samples_generated must be of type boolean')

    def validate_number_of_samples_if_generated(self):
        if not type(self.number_of_samples_if_generated) is int:
            raise Exception('number_of_samples_if_generated must be of type int')
        if self.number_of_samples_if_generated < self.MINIMAL_NUMBER_OF_SAMPLES:
            raise Exception('number_of_samples_if_generated must be at least{}'.format(self.MINIMAL_NUMBER_OF_SAMPLES))

    def validate_number_of_dataset_if_not_generated(self):
        if not type(self.number_of_dataset_if_not_generated) is int:
            raise Exception('number_of_dataset_if_not_generated must be of type int')
        if self.number_of_dataset_if_not_generated < 0:
            raise Exception('number_of_dataset_if_not_generated must be positive')

    def validate_switch_columns_while_loading(self):
        if not type(self.switch_columns_while_loading) is bool:
            raise Exception('switch_columns_while_loading must be of type boolean')

    def validate_number_of_space_parts(self):
        if not type(self.number_of_space_parts) is int:
            raise Exception('number_of_space_parts must be of type int')
        if self.number_of_space_parts < 0:
            raise Exception('number_of_space_parts must be positive')

    def validate_number_of_classifiers(self):
        if not type(self.number_of_classifiers) is int:
            raise Exception('number_of_classifiers must be of type int')
        if self.number_of_classifiers <= 0:
            raise Exception('number_of_classifiers must be positive')

    def validate_number_of_best_classifiers(self):
        if not type(self.number_of_best_classifiers) is int:
            raise Exception('number_of_best_classifiers must be of type int')
        if self.number_of_best_classifiers < 0:
            raise Exception('number_of_best_classifiers must be positive')

    def validate_draw_color_plot(self):
        if not type(self.draw_color_plot) is bool:
            raise Exception('draw_color_plot must be of type boolean')

    def validate_write_computed_scores(self):
        if not type(self.write_computed_scores) is bool:
            raise Exception('write_computed_scores must be of type boolean')

    def validate_show_plots(self):
        if not type(self.show_plots) is bool:
            raise Exception('show_plots must be of type boolean')

    def validate_show_only_first_plot(self):
        if not type(self.show_only_first_plot) is bool:
            raise Exception('show_only_first_plot must be of type boolean')

    def validate_columns(self):
        if len(self.columns) == 0:
            return
        if len(self.columns) != 2:
            raise Exception('Columns must be vector of size 1 x 2')
        for el in self.columns:
            if not type(el) is int:
                raise Exception('Elements of columns must be of type int')
            if el < 0:
                raise Exception('Column number must be positive')
        if self.columns[0] == self.columns[1]:
            raise Exception('Elements of matrix must be different')

    def validate_is_validation_hard(self):
        if not type(self.is_validation_hard) is bool:
            raise Exception('is_validation_hard must be of type boolean')

    def validate_filename(self):
        if not type(self.filename) is str:
            raise Exception('filename must be of type str')

    def validate_generate_all_permutations(self):
        if not type(self.generate_all_permutations) is bool:
            raise Exception('generate_all_permutations must be of type bool')

    def validate_log_number(self):
        if not type(self.log_number) is int:
            raise Exception('log_number must be of type int')
