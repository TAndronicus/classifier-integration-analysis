import matplotlib.pyplot as plt
import ClassifLibrary
import FileHelper
from CompositionType import CompositionType
from NotEnoughSamplesError import NotEnoughSamplesError
import numpy as np
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
        raise e

    if show_plots:
        number_of_subplots = ClassifLibrary.determine_number_of_subplots(classif_data)
    else:
        number_of_subplots = 0

    permutations = ClassifLibrary.generate_permutations(classif_data)

    scores_pro_space_division_pro_nbest, mccs_pro_space_division_pro_nbest = \
        ClassifLibrary.initialize_list_of_lists(number_of_classifiers - 2), \
        ClassifLibrary.initialize_list_of_lists(number_of_classifiers - 2)

    for n_perm in range(len(permutations)):

        print('\n{}. iteration\n'.format(n_perm + 1))

        X_whole_train, y_whole_train, X_validation, y_validation, X_test, y_test = \
            ClassifLibrary.get_permutation(X_splitted, y_splitted, permutations[n_perm], classif_data)

        clfs, coefficients = \
            ClassifLibrary.train_classifiers(clfs, X_whole_train, y_whole_train, X, number_of_subplots, classif_data)

        for n_best in range(2, number_of_classifiers):
            scores_pro_space_division, mccs_pro_space_division = [], []
            classif_data.number_of_best_classifiers = n_best
            for i in range(len(space_division)):
                print('{}. space division: {}'.format(i, space_division[i]))
                classif_data.number_of_space_parts = space_division[i]
                scores, cumulated_scores = \
                    ClassifLibrary.test_classifiers(clfs, X_validation, y_validation, coefficients, classif_data)

                confusion_matrices = ClassifLibrary.compute_confusion_matrix(clfs, X_test, y_test)

                mv_conf_mat, mv_score = ClassifLibrary.prepare_majority_voting(clfs, X_test, y_test)
                confusion_matrices.append(mv_conf_mat)
                cumulated_scores.append(mv_score)

                if type_of_composition == CompositionType.MEAN:
                    scores, i_score, i_conf_mat = \
                        ClassifLibrary.prepare_composite_mean_classifier(X_test,
                                                                         y_test,
                                                                         X,
                                                                         coefficients,
                                                                         scores,
                                                                         number_of_subplots,
                                                                         i,
                                                                         classif_data)
                elif type_of_composition == CompositionType.MEDIAN:
                    scores, i_score, i_conf_mat = \
                        ClassifLibrary.prepare_composite_median_classifier(X_test,
                                                                           y_test,
                                                                           X,
                                                                           coefficients,
                                                                           scores,
                                                                           number_of_subplots,
                                                                           i,
                                                                           classif_data)

                confusion_matrices.append(i_conf_mat)
                cumulated_scores.append(i_score)
                mccs = ClassifLibrary.compute_mccs(confusion_matrices)
                for m in mccs:
                    if abs(m) > 1:
                        print('Fuckup1')
                scores_pro_space_division.append(cumulated_scores)
                mccs_pro_space_division.append(mccs)

                ClassifLibrary.print_scores_conf_mats_mcc_pro_classif_pro_subspace(scores,
                                                                                   cumulated_scores,
                                                                                   confusion_matrices,
                                                                                   mccs)

            if show_plots:
                try:
                    plt.show()
                except AttributeError:
                    indicate_insufficient_samples()

            if show_only_first_plot:
                show_plots = False
                classif_data.show_plots = False
                classif_data.draw_color_plot = False

            scores_pro_space_division_pro_nbest[n_best - 2].append(scores_pro_space_division)
            mccs_pro_space_division_pro_nbest[n_best - 2].append(mccs_pro_space_division)

    print('\n#####\nOverall results_pro_division after {} iterations:'.format(len(permutations)))
    mean_scores, mean_mccs = \
        np.mean(scores_pro_space_division_pro_nbest, 1), np.mean(mccs_pro_space_division_pro_nbest, 1)
    std_scores, std_mccs = np.std(scores_pro_space_division_pro_nbest, 1), np.std(mccs_pro_space_division_pro_nbest, 1)

    k = 0
    classif_iter = 2
    list_of_results_pro_selection = []
    for mean_score_pro_selection, mean_mcc_pro_selection, std_score_pro_selection, std_mcc_pro_selection \
            in zip(mean_scores, mean_mccs, std_scores, std_mccs):
        print('###  {} best classifiers  ###'.format(classif_iter))
        list_of_results_pro_space_division, space_iter = [], 0
        for mean_score_pro_space_division, mean_mcc_pro_space_division, \
            std_score_pro_space_division, std_mcc_pro_space_division \
                in zip(mean_score_pro_selection, mean_mcc_pro_selection,
                       std_score_pro_selection, std_mcc_pro_selection):
            print('###  {} subspaces  ###'.format(space_division[space_iter]))
            for m in mean_mcc_pro_space_division:
                if abs(m) > 1:
                    print('Fuckup2')
            ClassifLibrary.print_permutation_results(mean_score_pro_space_division, mean_mcc_pro_space_division)
            res = ClassifLibrary.prepare_result_object(mean_score_pro_space_division,
                                                       mean_mcc_pro_space_division,
                                                       std_score_pro_space_division,
                                                       std_mcc_pro_space_division)
            if logging_intermediate_results and not bagging:
                FileHelper.save_intermediate_results(mean_score_pro_space_division,
                                                     mean_mcc_pro_space_division,
                                                     k,
                                                     classif_data)
            list_of_results_pro_space_division.append(res)
            k += 1
            space_iter += 1
        list_of_results_pro_selection.append(list_of_results_pro_space_division)
        classif_iter += 1

    if logging_to_file:
        disable_logging_to_file()
    return list_of_results_pro_selection
