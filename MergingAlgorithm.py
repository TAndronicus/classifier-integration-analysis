import matplotlib.pyplot as plt
import ClassifLibrary


def run(classif_data = ClassifLibrary.ClassifierData()):
    """Invokes merging algorithm for classification data

    :param classif_data: ClassifLibrary.ClassifierData
    :return: mv_score, merged_score, mv_mcc, merged_mcc
    """
    classif_data.validate()
    show_plots = classif_data.show_plots
    show_only_first_plot = classif_data.show_only_first_plot

    clfs = ClassifLibrary.initialize_classifiers(classif_data)

    X, y = ClassifLibrary.prepare_raw_data(classif_data)

    X_splitted, y_splitted = ClassifLibrary.split_sorted_unitary(X, y, classif_data)

    if show_plots:
        number_of_subplots = ClassifLibrary.determine_number_of_subplots(classif_data)
    else:
        number_of_subplots = 0

    number_of_permutations = 1

    score_pro_permutation, mccs_pro_permutation = [], []
    permutations = ClassifLibrary.generate_permutations(classif_data)
    for tup in permutations:

        print('\n{}. iteration\n'.format(number_of_permutations))

        X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
            ClassifLibrary.get_permutation(X_splitted, y_splitted, tup, classif_data)

        clfs, coefficients = \
            ClassifLibrary.train_classifiers(clfs, X_whole_train, y_whole_train, X, number_of_subplots, classif_data)

        scores, cumulated_scores = ClassifLibrary.test_classifiers(clfs, X_validation, y_validation, X, coefficients,
                                                                   classif_data)

        confusion_matrices = ClassifLibrary.compute_confusion_matrix(clfs, X_test, y_test)

        mv_conf_mat, mv_score = ClassifLibrary.prepare_majority_voting(clfs, X_test, y_test)
        confusion_matrices.append(mv_conf_mat)
        cumulated_scores.append(mv_score)

        scores, cumulated_score, conf_mat = \
            ClassifLibrary.prepare_composite_classifier(X_test, y_test, X, coefficients, scores, number_of_subplots,
                                                        classif_data)

        confusion_matrices.append(conf_mat)
        cumulated_scores.append(cumulated_score)
        mccs = ClassifLibrary.compute_mcc(confusion_matrices)
        score_pro_permutation.append(cumulated_scores)
        mccs_pro_permutation.append(mccs)

        ClassifLibrary.print_scores_conf_mats_mcc_pro_classif_pro_subspace(scores, cumulated_scores,
                                                                           confusion_matrices, mccs)

        number_of_permutations += 1

        if show_plots:
            plt.show()

        if show_only_first_plot:
            show_plots = False  # Convenience
            classif_data.show_plots = False
            classif_data.draw_color_plot = False
    number_of_permutations -= 1

    print('\n#####\nOverall results after {} iterations:'.format(number_of_permutations))
    overall_scores, overall_mcc = ClassifLibrary.get_permutation_results(score_pro_permutation, mccs_pro_permutation)
    ClassifLibrary.print_permutation_results(overall_scores, overall_mcc)
    print('\n#####\n')
    return overall_scores[-2], overall_scores[-1], overall_mcc[-2], overall_mcc[-1]
