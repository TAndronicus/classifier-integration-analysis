import xlwt
from ClassifierData import ClassifierData


def save_merging_results_one_space_division(filenames: [], results: [], result_filename: str = 'results//Results.xls',
                                            sheetname: str = 'Result'):
    """Saves results of merging algorithm for one space division and one number of base classifier

    :param filenames: names of files being analysed
    :param results: resulting matrix
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :return:
    """
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 0, "filename")
    sheet.write(0, 1, "majority voting score")
    sheet.write(0, 2, "integrated classifier score")
    sheet.write(0, 3, "majority voting matthews correlation coefficient")
    sheet.write(0, 4, "integrated classifier matthews correlation coefficient")
    for i in range(len(filenames)):
        sheet.write(i + 1, 0, filenames[i])
        for j in range(len(results[i])):
            sheet.write(i + 1, j + 1, results[i][j])
    workbook.save(result_filename)


def save_merging_results_pro_space_division(filenames: [], results_pro_space_division: [], space_division: [],
                                            result_filename: str = 'results//Results.xls', sheetname: str = 'Result'):
    """Saves results of merging algorithm for one number of base classifier

    :param filenames: names of files being analysed
    :param results_pro_space_division: matrix of results pro space division
    :param space_division: array of space divisions
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :return:
    """
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 0, "subspaces")
    sheet.write(1, 0, "filename")
    for i in range(len(space_division)):
        sheet.write(0, 4 * i + 1, str(space_division[i]))
        sheet.write(1, 4 * i + 1, "mv_s")
        sheet.write(1, 4 * i + 2, "i_s")
        sheet.write(1, 4 * i + 3, "mv_mcc")
        sheet.write(1, 4 * (i + 1), "i_mcc")
    for i in range(len(filenames)):
        sheet.write(i + 2, 0, filenames[i])
        for j in range(len(space_division)):
            for k in range(len(results_pro_space_division[j][i])):
                sheet.write(i + 2, 4 * j + k + 1, results_pro_space_division[j][i][k])
    workbook.save(result_filename)


def save_merging_results_pro_space_division_pro_base_classif(filenames: [],
                                                             results_pro_space_division_pro_base_classif: [],
                                                             numbers_of_base_classifiers: [], space_division: [],
                                                             result_filename: str = 'results//Results.xls',
                                                             sheetname: str = 'Result'):
    """Saves results of merging algorithm

    :param filenames: names of files being analysed
    :param results_pro_space_division_pro_base_classif: matrix of results pro base classifier
    :param numbers_of_base_classifiers: array of numbers of base classifiers
    :param space_division: array of space divisions
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :return:
    """
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 1, "subspaces")
    sheet.write(1, 0, "classifiers")
    sheet.write(1, 1, "filename")
    for j in range(len(space_division)):
        sheet.write(0, 4 * j + 2, str(space_division[j]))
        sheet.write(1, 4 * j + 2, "mv_s")
        sheet.write(1, 4 * j + 3, "i_s")
        sheet.write(1, 4 * (j + 1), "mv_mcc")
        sheet.write(1, 4 * (j + 1) + 1, "i_mcc")
    for j in range(len(numbers_of_base_classifiers)):
        sheet.write(len(filenames) * j + 2, 0, str(numbers_of_base_classifiers[j]))
    for i in range(len(numbers_of_base_classifiers)):
        for j in range(len(filenames)):
            sheet.write(i * len(filenames) + j + 2, 1, filenames[j])
            for k in range(len(space_division)):
                for l in range(len(results_pro_space_division_pro_base_classif[i][k][j])):
                    sheet.write(i * len(filenames) + j + 2, 4 * k + l + 2,
                                results_pro_space_division_pro_base_classif[i][k][j][l])
    workbook.save(result_filename)


def save_merging_results_pro_space_division_pro_base_classif_with_classif_data(filenames: [],
                                                                               res: [],
                                                                               numbers_of_base_classifiers: [],
                                                                               space_division: [],
                                                                               result_filename: str =
                                                                               'results//Results.xls',
                                                                               sheetname: str = 'Result',
                                                                               classifier_data: ClassifierData =
                                                                               ClassifierData()):
    """Saves results of merging algorithm

    :param filenames: names of files being analysed
    :param res: result objects pro base classifiers pro space division pro file
    :param numbers_of_base_classifiers: array of numbers of base classifiers
    :param space_division: array of space divisions
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :param classifier_data: parameter object
    :return:
    """
    results_pro_space_division_pro_base_classif = generate_partial_result_matrix(res)
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 1, "subspaces")
    sheet.write(1, 0, "classifiers")
    sheet.write(1, 1, "filename")
    for j in range(len(space_division)):
        sheet.write(0, 4 * j + 2, str(space_division[j]))
        sheet.write(1, 4 * j + 2, "mv_s")
        sheet.write(1, 4 * j + 3, "i_s")
        sheet.write(1, 4 * (j + 1), "mv_mcc")
        sheet.write(1, 4 * (j + 1) + 1, "i_mcc")
    for j in range(len(numbers_of_base_classifiers)):
        sheet.write(len(filenames) * j + 2, 0, str(numbers_of_base_classifiers[j]))
    for i in range(len(numbers_of_base_classifiers)):
        for j in range(len(filenames)):
            sheet.write(i * len(filenames) + j + 2, 1, filenames[j])
            for k in range(len(space_division)):
                for l in range(len(results_pro_space_division_pro_base_classif[i][k][j])):
                    sheet.write(i * len(filenames) + j + 2, 4 * k + l + 2,
                                results_pro_space_division_pro_base_classif[i][k][j][l])
    output_data = {'type_of_classifier': classifier_data.type_of_classifier.value,
                   'are_samples_generated': str(classifier_data.are_samples_generated),
                   'number_of_samples_if_generated': classifier_data.number_of_samples_if_generated,
                   'number_of_dataset_if_not_generated': classifier_data.number_of_dataset_if_not_generated,
                   'switch_columns_while_loading': str(classifier_data.switch_columns_while_loading),
                   'number_of_best_classifiers': classifier_data.number_of_best_classifiers,
                   'columns': str(classifier_data.columns),
                   'is_validation_hard': str(classifier_data.is_validation_hard),
                   'generate_all_permutations': str(classifier_data.generate_all_permutations),
                   'bagging': str(classifier_data.bagging),
                   'type_of_composition': classifier_data.type_of_composition.value}
    last_row = 1 + len(filenames) * len(numbers_of_base_classifiers)
    for entry_name in output_data:
        last_row += 1
        sheet.write(last_row, 0, entry_name)
        sheet.write(last_row, 1, output_data.get(entry_name))
    workbook.save(result_filename)


def generate_partial_result_matrix(res):
    """Converts result object into partial matrix

    :param res: IntegrRes
    :return: overall_results: []
    """
    overall_results = []
    for result_pro_classifier in res:
        mats_pro_classifier = []
        for result_pro_space_division in result_pro_classifier:
            mats_pro_space_division = []
            for result_pro_file in result_pro_space_division:
                mat = [result_pro_file.mv_score, result_pro_file.i_score, result_pro_file.mv_mcc, result_pro_file.i_mcc]
                mats_pro_space_division.append(mat)
            mats_pro_classifier.append(mats_pro_space_division)
        overall_results.append(mats_pro_classifier)
    return overall_results
