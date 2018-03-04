from ClfType import ClfType


class ClassifierData():
    """Clas to produce parameter object for classification

    """

    MINIMAL_NUMBER_OF_SAMPLES = 10

    def __init__(self, type_of_classifier = ClfType.LINEAR, are_samples_generated = True,
                 number_of_samples_if_generated = 1000, number_of_dataset_if_not_generated = 12,
                 switch_columns_while_loading = False, plot_mesh_step_size = .2, number_of_space_parts = 5,
                 number_of_classifiers = 3, number_of_best_classifiers = 2, draw_color_plot = False,
                 write_computed_scores = False, show_plots = False, columns = [0, 1], is_validation_hard = False,
                 filename = 'new-datasets.xlsx'):
        self.type_of_classifier = type_of_classifier
        self.are_samples_generated = are_samples_generated
        self.number_of_samples_if_generated = number_of_samples_if_generated
        self.number_of_dataset_if_not_generated = number_of_dataset_if_not_generated
        self.switch_columns_while_loading = switch_columns_while_loading
        self.plot_mesh_step_size = plot_mesh_step_size
        self.number_of_space_parts = number_of_space_parts
        self.number_of_classifiers = number_of_classifiers
        self.number_of_best_classifiers = number_of_best_classifiers
        self.draw_color_plot = draw_color_plot
        self.write_computed_scores = write_computed_scores
        self.show_plots = show_plots
        self.columns = columns
        self.is_validation_hard = is_validation_hard
        self.filename = filename

    def validate(self):
        self.validate_type_of_classifier()
        self.validate_are_samples_generated()
        self.validate_number_of_samples_if_generated()
        self.validate_number_of_dataset_if_not_generated()
        self.validate_switch_columns_while_loading()
        self.validate_plot_mesh_step_size()
        self.validate_number_of_space_parts()
        self.validate_number_of_classifiers()
        self.validate_number_of_best_classifiers()
        self.validate_draw_color_plot()
        self.validate_write_computed_scores()
        self.validate_show_plots()
        self.validate_columns()
        self.validate_is_validation_hard()
        self.validate_filename()

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

    def validate_plot_mesh_step_size(self):
        if not type(self.plot_mesh_step_size) is float:
            raise Exception('plot_mesh_step_size must be of type float')
        if self.plot_mesh_step_size <= 0:
            raise Exception('plot_mesh_step_size must be positive')

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
