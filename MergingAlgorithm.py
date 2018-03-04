import matplotlib.pyplot as plt
import ClassifLibrary

type_of_classifier = ClassifLibrary.ClfType.LINEAR
are_samples_generated = False
number_of_samples_if_generated = 10000
number_of_dataset_if_not_generated = 14
columns = [0, 1]
plot_mesh_step_size = .2
number_of_space_parts = 5
number_of_classifiers = 3
number_of_best_classifiers = number_of_classifiers - 1
draw_color_plot = False
write_computed_scores = False
show_plots = False
is_validation_hard = False
filename = 'new-datasets.xlsx'

classifier_data = \
    ClassifLibrary.ClassifierData(type_of_classifier = type_of_classifier, are_samples_generated = are_samples_generated,
                                  number_of_samples_if_generated = number_of_samples_if_generated,
                                  number_of_dataset_if_not_generated = number_of_dataset_if_not_generated,
                                  plot_mesh_step_size = plot_mesh_step_size, number_of_space_parts = number_of_space_parts,
                                  number_of_classifiers = number_of_classifiers,
                                  number_of_best_classifiers = number_of_best_classifiers,
                                  draw_color_plot = draw_color_plot, write_computed_scores = write_computed_scores,
                                  show_plots = show_plots, columns = columns, is_validation_hard = is_validation_hard,
                                  filename = filename)

clfs = ClassifLibrary.initialize_classifiers(classifier_data)

X, y = ClassifLibrary.prepare_raw_data(classifier_data)

X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
    ClassifLibrary.split_sorted_samples(X, y, classifier_data)

if show_plots:
    xx, yy, x_min_plot, x_max_plot = ClassifLibrary.get_plot_data(X, classifier_data)
    number_of_subplots = ClassifLibrary.determine_number_of_subplots(classifier_data)
else:
    number_of_subplots = 0

number_of_permutations = 0

score_pro_permutation = []
while True:
    print("\n{}. iteration\n".format(number_of_permutations))
    clfs, coefficients = \
        ClassifLibrary.train_classifiers(clfs, X_whole_train, y_whole_train, X, number_of_subplots, classifier_data)

    scores, cumulated_scores = ClassifLibrary.test_classifiers(clfs, X_validation, y_validation, X, coefficients,
                                                               classifier_data)

    confusion_matrices = ClassifLibrary.compute_confusion_matrix(clfs, X_test, y_test)

    mv_conf_mat, mv_score = ClassifLibrary.prepare_majority_voting(clfs, X_test, y_test)
    confusion_matrices.append(mv_conf_mat)
    cumulated_scores.append(mv_score)

    scores, cumulated_score, conf_mat = \
        ClassifLibrary.prepare_composite_classifier(X_test, y_test, X, coefficients, scores, number_of_subplots,
                                                    classifier_data)

    confusion_matrices.append(conf_mat)
    cumulated_scores.append(cumulated_score)
    score_pro_permutation.append(cumulated_scores)

    mcc = ClassifLibrary

    ClassifLibrary.print_results_with_conf_mats(scores, cumulated_scores, confusion_matrices)

    X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
        ClassifLibrary.generate_permutation(X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test)
    number_of_permutations += 1

    if show_plots:
        plt.show()

    if number_of_permutations == number_of_classifiers + 2:
        break

print("\n\nOverall results:")
ClassifLibrary.print_permutation_results(score_pro_permutation)
