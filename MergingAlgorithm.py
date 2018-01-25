import numpy as np
import matplotlib.pyplot as plt
import MyLibrary

type_of_classifier = MyLibrary.ClfType.LINEAR
are_samples_generated = True
number_of_samples_if_generated = 100
number_of_dataset_if_not_generated = 12
plot_mesh_step_size = .2
number_of_space_parts = 5
number_of_classifiers = 3
number_of_best_classifiers = number_of_classifiers - 1
draw_color_plot = False
write_computed_scores = False
show_plots = False

clfs = MyLibrary.initialize_classifiers(number_of_classifiers, type_of_classifier)

X, y = MyLibrary.prepare_raw_data(are_samples_generated, number_of_samples_if_generated, number_of_dataset_if_not_generated, number_of_classifiers,
                                  number_of_space_parts)


X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = MyLibrary.split_sorted_samples(X, y, number_of_classifiers, number_of_space_parts)

xx, yy, x_min_plot, x_max_plot = MyLibrary.get_plot_data(X, plot_mesh_step_size)
number_of_subplots = MyLibrary.determine_number_of_subplots(draw_color_plot, number_of_classifiers)

number_of_permutations = 0

score_pro_permutation = []
while(True):
    clfs, coefficients = MyLibrary.train_classifiers(clfs, X_whole_train, y_whole_train, type_of_classifier,
                                                 number_of_subplots, X, plot_mesh_step_size, draw_color_plot)

    scores, cumulated_scores = MyLibrary.test_classifiers(clfs, X_validation, y_validation, X, coefficients, number_of_space_parts, write_computed_scores)

    scores, cumulated_score = MyLibrary.prepare_composite_classifier(X_test, y_test, X, number_of_best_classifiers, coefficients, scores, number_of_subplots, number_of_space_parts, number_of_classifiers, plot_mesh_step_size)

    cumulated_scores.append(cumulated_score)
    score_pro_permutation.append(cumulated_scores)

    MyLibrary.print_results_with_cumulated_score(scores, cumulated_scores)

    X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = MyLibrary.generate_permutation(X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test)
    number_of_permutations += 1

    if show_plots:
        plt.show()

    if(number_of_permutations == number_of_classifiers + 2):
        break

MyLibrary.print_permutation_results(score_pro_permutation)