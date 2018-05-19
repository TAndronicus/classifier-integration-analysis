import xlrd
import xlwt
from AdvIntegrRes import AdvIntegrRes
from ClassifierData import ClassifierData
from CompositionType import CompositionType
from datetime import datetime
import numpy as np
import os


FILENAMES = ['biodeg.scsv',
             'bupa.dat',
             'cryotherapy.xlsx',
             'data_banknote_authentication.csv',
             'haberman.dat',
             'ionosphere.dat',
             'meter_a.tsv',
             'pop_failures.tsv',
             'seismic_bumps.dat',
             'twonorm.dat',
             'wdbc.dat',
             'wisconsin.dat']


def prepare_filenames(filenames_raw: []):
    """Prepares array with right filenames based on array with first parts of them

    :param filenames_raw: []
    :return: []
    """
    filenames = []
    for filename_raw in filenames_raw:
        try:
            filename = get_full_filename(filename_raw)
            filenames.append(filename)
        except FileNotFoundError as e:
            raise FileNotFoundError(e.args[0] + ': filename = ' + filename_raw)
    return filenames


def get_full_filename(filename_raw: str):
    """Returns whole filename basen on the first part

    :param filename_raw: str
    :return: str
    """
    was_found = False
    for FILENAME in FILENAMES:
        if FILENAME.startswith(filename_raw):
            if was_found:
                raise FileNotFoundError('Name of file wrong or ambiguous')
            else:
                filename = FILENAME
                was_found = True
    if was_found:
        return filename
    else:
        raise FileNotFoundError('Name of file not found')


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


def save_res_objects_pro_space_division_pro_base_classif_with_classif_data(filenames: [],
                                                                           results_pro_space_division_pro_base_classif:
                                                                           [],
                                                                           numbers_of_base_classifiers: [],
                                                                           result_filename: str =
                                                                           'results//Results.xls',
                                                                           sheetname: str = 'Result',
                                                                           classifier_data: ClassifierData =
                                                                           ClassifierData()):
    """Saves results of merging algorithm

    :param filenames: names of files being analysed
    :param results_pro_space_division_pro_base_classif: result objects pro base classifiers pro space division pro file
    :param numbers_of_base_classifiers: array of numbers of base classifiers
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :param classifier_data: parameter object
    :return:
    """
    space_division = classifier_data.space_division
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 1, "subspaces")
    sheet.write(1, 0, "classifiers")
    sheet.write(1, 1, "filename")
    for j in range(len(space_division)):
        sheet.write(0, 8 * j + 2, str(space_division[j]))
        sheet.write(1, 8 * j + 2, "mv_score")
        sheet.write(1, 8 * j + 3, "mv_score_std")
        sheet.write(1, 8 * j + 4, "mv_mcc")
        sheet.write(1, 8 * j + 5, "mv_mcc_std")
        sheet.write(1, 8 * j + 6, "i_score")
        sheet.write(1, 8 * j + 7, "i_score_std")
        sheet.write(1, 8 * j + 8, "i_mcc")
        sheet.write(1, 8 * j + 9, "i_mcc_std")
    for j in range(len(numbers_of_base_classifiers)):
        sheet.write(len(filenames) * j + 2, 0, str(numbers_of_base_classifiers[j]))
    for i in range(len(numbers_of_base_classifiers)):
        for j in range(len(filenames)):
            sheet.write(i * len(filenames) + j + 2, 1, filenames[j])
            for k in range(len(space_division)):
                res = results_pro_space_division_pro_base_classif[i][j][k]
                sheet.write(i * len(filenames) + j + 2, 8 * k + 2, res.mv_score)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 3, res.mv_score_std)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 4, res.mv_mcc)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 5, res.mv_mcc_std)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 6, res.i_score)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 7, res.i_score_std)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 8, res.i_mcc)
                sheet.write(i * len(filenames) + j + 2, 8 * k + 9, res.i_mcc_std)
    output_data = {'type_of_classifier': classifier_data.type_of_classifier.value,
                   'are_samples_generated': str(classifier_data.are_samples_generated),
                   'number_of_samples_if_generated': classifier_data.number_of_samples_if_generated,
                   'number_of_dataset_if_not_generated': classifier_data.number_of_dataset_if_not_generated,
                   'number_of_best_classifiers': classifier_data.number_of_best_classifiers,
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


def save_res_objects_pro_space_division_pro_base_classif_with_classif_data_name(filenames: [],
                                                                                results_pro_space_division_pro_base_classif:
                                                                                [],
                                                                                numbers_of_base_classifiers: [],
                                                                                results_directory_relative: str = 'results',
                                                                                sheetname: str = 'Result',
                                                                                classifier_data: ClassifierData =
                                                                                ClassifierData()):
    """Saves results of merging algorithm

    :param filenames: names of files being analysed
    :param results_pro_space_division_pro_base_classif: result objects pro base classifiers pro space division pro file
    :param numbers_of_base_classifiers: array of numbers of base classifiers
    :param result_filename: filename to write results to
    :param sheetname: sheetname to write results to
    :param classifier_data: parameter object
    :return:
    """
    space_division = classifier_data.space_division
    workbook = xlwt.Workbook()
    workbook.add_sheet(sheetname)
    sheet = workbook.get_sheet(sheetname)
    sheet.write(0, 1, "subspaces")
    sheet.write(1, 0, "selected classifiers")
    sheet.write(1, 1, "filename")
    for j in range(len(space_division)):
        sheet.write(0, 8 * j + 2, str(space_division[j]))
        sheet.write(1, 8 * j + 2, "mv_score")
        sheet.write(1, 8 * j + 3, "mv_score_std")
        sheet.write(1, 8 * j + 4, "mv_mcc")
        sheet.write(1, 8 * j + 5, "mv_mcc_std")
        sheet.write(1, 8 * j + 6, "i_score")
        sheet.write(1, 8 * j + 7, "i_score_std")
        sheet.write(1, 8 * j + 8, "i_mcc")
        sheet.write(1, 8 * j + 9, "i_mcc_std")
    for j in range(2, numbers_of_base_classifiers):
        sheet.write(len(filenames) * (j - 2) + 2, 0, str(j))
    for i in range(len(filenames)):
        for j in range(numbers_of_base_classifiers - 2):
            sheet.write(j * len(filenames) + i + 2, 1, filenames[i])
            for k in range(len(space_division)):
                res = results_pro_space_division_pro_base_classif[i][j][k]
                sheet.write(j * len(filenames) + i + 2, 8 * k + 2, res.mv_score)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 3, res.mv_score_std)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 4, res.mv_mcc)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 5, res.mv_mcc_std)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 6, res.i_score)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 7, res.i_score_std)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 8, res.i_mcc)
                sheet.write(j * len(filenames) + i + 2, 8 * k + 9, res.i_mcc_std)
    output_data = {'type_of_classifier': classifier_data.type_of_classifier.value,
                   'are_samples_generated': str(classifier_data.are_samples_generated),
                   'number_of_samples_if_generated': classifier_data.number_of_samples_if_generated,
                   'number_of_dataset_if_not_generated': classifier_data.number_of_dataset_if_not_generated,
                   'is_validation_hard': str(classifier_data.is_validation_hard),
                   'generate_all_permutations': str(classifier_data.generate_all_permutations),
                   'bagging': str(classifier_data.bagging),
                   'type_of_composition': classifier_data.type_of_composition.value,
                   'timestamp': str(datetime.now())}
    last_row = 1 + len(filenames) * (numbers_of_base_classifiers - 2)
    for entry_name in output_data:
        last_row += 1
        sheet.write(last_row, 0, entry_name)
        sheet.write(last_row, 1, output_data.get(entry_name))
    result_filename = determine_filename(results_directory_relative, classifier_data)
    workbook.save(result_filename)


def determine_filename(results_directory_relative: str = 'results', classifier_data: ClassifierData = ClassifierData()):
    if classifier_data.bagging:
        bagging_indicator = str(1)
    else:
        bagging_indicator = str(0)
    if classifier_data.type_of_composition == CompositionType.MEAN:
        integration_indicator = str(0)
    else:
        integration_indicator = str(1)
    filename = 'n_' + str(classifier_data.number_of_classifiers) + \
               '_b_' + bagging_indicator + \
               '_i_' + integration_indicator
    if os.path.isfile(results_directory_relative + '//' + filename + '.xls'):
        log_number = 1
        while True:
            if not os.path.isfile(results_directory_relative + '//' + filename + '_v_' + str(log_number) + '.xls'):
                break
            log_number += 1
        filename = filename + '_v_' + str(log_number)
    return results_directory_relative + '//' + filename + '.xls'


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


def save_intermediate_results(score: [], mcc: [], i: int, classifier_data: ClassifierData = ClassifierData()):
    """Saves intermediate results

    :param score: []
    :param mcc: []
    :param i: int
    :param classifier_data: ClassifierData
    :return:
    """
    filename = classifier_data.filename
    number_of_classifiers = classifier_data.number_of_classifiers
    space_division = classifier_data.space_division
    data_to_save = {
        'score': score,
        'mcc': mcc
    }
    results_directory_relative = 'intermediate_results'

    results_directory_absolute = os.path.join(os.path.dirname(__file__), results_directory_relative)
    try:
        os.makedirs(results_directory_absolute)
        print('Created results directory: ', results_directory_absolute)
    except FileExistsError:
        pass

    workbook = xlwt.Workbook()
    for key, value in data_to_save.items():
        workbook.add_sheet(key)
        sheet = workbook.get_sheet(key)
        for row in range(len(value)):
            for col in range(len(value[row])):
                sheet.write(row, col, value[row][col])
    workbook.save(results_directory_relative + '//' + filename.split('.')[0].split('//')[1] + '_c_' + str(
        number_of_classifiers) + '_s_' + str(space_division[i]) + '.xls')


def read_objects_from_file(res_filename: str, n_class: int, bagging: int, i_meth: int):
    """Reads objects from result files

    :param res_filename: str, one of: 'biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx', 'data_banknote_authentication.csv',
                 'haberman.dat', 'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv', 'seismic_bumps.dat',
                 'twonorm.dat', 'wdbc.dat', 'wisconsin.dat'
    :param n_class: int
    :param bagging: int
    :param i_meth: int
    :return:
    """
    filenames = ['biodeg.scsv', 'bupa.dat', 'cryotherapy.xlsx', 'data_banknote_authentication.csv',
                 'haberman.dat', 'ionosphere.dat', 'meter_a.tsv', 'pop_failures.tsv', 'seismic_bumps.dat',
                 'twonorm.dat', 'wdbc.dat', 'wisconsin.dat']
    file = xlrd.open_workbook(res_filename)
    sheet = file.sheet_by_index(0)
    result_objects = []
    for line_num in range(sheet.nrows):
        line = sheet.row(line_num)
        if line[1].value in filenames:
            n_best_line = line_num
            while True:
                if sheet.cell(n_best_line, 0).ctype == 1:
                    n_best = int(sheet.cell(n_best_line, 0).value)
                    break
                n_best_line -= 1
            for n_subspace in range(int((sheet.ncols - 2) / 8)):
                space_parts = int(sheet.cell(0, 2 + n_subspace * 8).value)
                mv_score = sheet.cell(line_num, 8 * n_subspace + 2).value
                mv_score_std = sheet.cell(line_num, 8 * n_subspace + 3).value
                mv_mcc = sheet.cell(line_num, 8 * n_subspace + 4).value
                mv_mcc_std = sheet.cell(line_num, 8 * n_subspace + 5).value
                i_score = sheet.cell(line_num, 8 * n_subspace + 6).value
                i_score_std = sheet.cell(line_num, 8 * n_subspace + 7).value
                i_mcc = sheet.cell(line_num, 8 * n_subspace + 8).value
                i_mcc_std = sheet.cell(line_num, 8 * n_subspace + 9).value
                res_obj = AdvIntegrRes(mv_score, mv_score_std, mv_mcc, mv_mcc_std, i_score, i_score_std, i_mcc, i_mcc_std, n_class, n_best, i_meth, bagging, space_parts, line[1].value)
                result_objects.append(res_obj)
    return result_objects
