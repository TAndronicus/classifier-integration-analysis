import matplotlib.pyplot as plt
import ClassifLibrary
import FileHelper
from CompositionType import CompositionType
from NotEnoughSamplesError import NotEnoughSamplesError
import sys


def enable_logging_to_file(log_number):
    sys.stdout = open('results//integration' + str(log_number) + '.log', 'a')


def disable_logging_to_file():
    sys.stdout = sys.__stdout__


def indicate_insufficient_samples(e: NotEnoughSamplesError = NotEnoughSamplesError('Not enough samples for plot'),
                                  classifier_data: ClassifLibrary.ClassifierData = ClassifLibrary.ClassifierData()):
    print('\n#####\n')
    print('Not enough samples, raising error (filename = {}, number of classifiers = {}, number of subspaces = {})'
          .format(classifier_data.filename, classifier_data.number_of_classifiers,
                  classifier_data.number_of_space_parts))
    print(e.args[0])
    print('\n#####\n')
    raise e


def run(classif_data = ClassifLibrary.ClassifierData()):
    """Invokes merging algorithm for classification data

    :param classif_data: ClassifLibrary.ClassifierData
    :return: mv_score, merged_score, mv_mcc, merged_mcc
    """
    log_number = classif_data.log_number
    logging_to_file = classif_data.logging_to_file
    logging_intermediate_results = classif_data.logging_intermediate_results

    bagging = classif_data.bagging
    type_of_composition = classif_data.type_of_composition
    space_division = classif_data.space_division
    number_of_classifiers = classif_data.number_of_classifiers

    if logging_to_file:
        enable_logging_to_file(log_number)
    classif_data.validate()
    show_plots = classif_data.show_plots
    show_only_first_plot = classif_data.show_only_first_plot

    clfs = ClassifLibrary.initialize_classifiers(classif_data)

    X, y = ClassifLibrary.prepare_raw_data(classif_data)

    try:
        if bagging:
            X_splitted, y_splitted = ClassifLibrary.split_sorted_unitary_bagging(X, y, classif_data)
        else:
            X_splitted, y_splitted = ClassifLibrary.split_sorted_unitary(X, y, classif_data)
    except NotEnoughSamplesError as e:
        X_splitted, y_splitted = [], []
        indicate_insufficient_samples(e, classif_data)

    if show_plots:
        number_of_subplots = ClassifLibrary.determine_number_of_subplots(classif_data)
    else:
        number_of_subplots = 0

    number_of_permutations = 1

    score_pro_space_division, mccs_pro_space_division = \
        ClassifLibrary.initialize_list_of_lists(len(classif_data.space_division)), \
        ClassifLibrary.initialize_list_of_lists(len(classif_data.space_division))
    score_pro_space_division_pro_permutation_pro_nbest, mccs_pro_space_division_pro_permutation_pro_nbest = [], []

    permutations = ClassifLibrary.generate_permutations(classif_data)

    for tup in permutations:

        print('\n{}. iteration\n'.format(number_of_permutations))

        X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
            ClassifLibrary.get_permutation(X_splitted, y_splitted, tup, classif_data)

        clfs, coefficients = \
            ClassifLibrary.train_classifiers(clfs, X_whole_train, y_whole_train, X, number_of_subplots, classif_data)

        for nbest in range(2, number_of_classifiers):
            classif_data.number_of_best_classifiers = nbest
            for i in range(len(space_division)):
                print('{}. space division: {}'.format(i, space_division[i]))
                classif_data.number_of_space_parts = space_division[i]
                scores, cumulated_scores = ClassifLibrary.test_classifiers(clfs, X_validation, y_validation, coefficients,
                                                                       classif_data)

                confusion_matrices = ClassifLibrary.compute_confusion_matrix(clfs, X_test, y_test)

                mv_conf_mat, mv_score = ClassifLibrary.prepare_majority_voting(clfs, X_test, y_test)
                confusion_matrices.append(mv_conf_mat)
                cumulated_scores.append(mv_score)

                if type_of_composition == CompositionType.MEAN:
                    scores, i_score, i_conf_mat = \
                        ClassifLibrary.prepare_composite_mean_classifier(X_test, y_test, X, coefficients, scores, number_of_subplots, i, classif_data)
                elif type_of_composition == CompositionType.MEDIAN:
                    scores, i_score, i_conf_mat = \
                        ClassifLibrary.prepare_composite_median_classifier(X_test, y_test, X, coefficients, scores, number_of_subplots, i, classif_data)

                confusion_matrices.append(i_conf_mat)
                cumulated_scores.append(i_score)
                mccs = ClassifLibrary.compute_mccs(confusion_matrices)
                score_pro_space_division[i].append(cumulated_scores)
                mccs_pro_space_division[i].append(mccs)

                ClassifLibrary.print_scores_conf_mats_mcc_pro_classif_pro_subspace(scores, cumulated_scores, confusion_matrices, mccs)

            number_of_permutations += 1

            if show_plots:
                try:
                    plt.show()
                except AttributeError:
                    indicate_insufficient_samples()

            if show_only_first_plot:
                show_plots = False
                classif_data.show_plots = False
                classif_data.draw_color_plot = False

            score_pro_space_division_pro_permutation_pro_nbest.append(score_pro_space_division)
            mccs_pro_space_division_pro_permutation_pro_nbest.append(mccs_pro_space_division)

    number_of_permutations -= 1

    print('\n#####\nOverall results_pro_division after {} iterations:'.format(number_of_permutations))
    k = 0
    list_of_results_pro_selection = []
    for i in range(len(score_pro_space_division_pro_permutation_pro_nbest)):
        score_pro_selection = score_pro_space_division_pro_permutation_pro_nbest[i]
        mcc_pro_selection = mccs_pro_space_division_pro_permutation_pro_nbest[i]
        list_of_results_pro_space_division = []
        n_div = 0
        for score_pro_permutation, mcc_pro_permutation in zip(score_pro_selection, mcc_pro_selection):
            print('Number of space divisions: ', space_division[n_div])
            overall_scores, overall_mccs = ClassifLibrary.get_permutation_results(score_pro_permutation, mcc_pro_permutation)
            overall_scores_std, overall_mccs_std = ClassifLibrary.get_permutation_stds(score_pro_permutation, mcc_pro_permutation)
            ClassifLibrary.print_permutation_results(overall_scores, overall_mccs)
            print('\n#####\n')
            res = ClassifLibrary.prepare_result_object(overall_scores, overall_mccs, overall_mccs_std, overall_mccs_std)
            if logging_intermediate_results and not(bagging):
                FileHelper.save_intermediate_results(score_pro_permutation, mcc_pro_permutation, k, classif_data)
            list_of_results_pro_space_division.append(res)
            n_div += 1
            k += 1
        list_of_results_pro_selection.append(list_of_results_pro_space_division)

    if logging_to_file:
        disable_logging_to_file()
    return list_of_results_pro_selection
